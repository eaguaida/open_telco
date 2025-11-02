import os
import sys
import subprocess
import json
import time
import uuid
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, List
from flask import Flask, render_template, request, jsonify, Response
from threading import Thread
from queue import Queue, Empty

app = Flask(__name__)

TASK_FILES: Dict[str, str] = {
    "telemath": "benchmarks/telemath/telemath.py",
    "teleqna": "benchmarks/teleqna/teleqna.py",
    "telelogs": "benchmarks/telelogs/telelogs.py",
}

TASK_ALIASES: Dict[str, str] = {
    "telecom_bench": "telemath",
    "teleqna_bench": "teleqna",
    "telelogs_bench": "telelogs",
}

# Try to find inspect command - check environment variable first, then common locations
INSPECT_CMD = os.environ.get("INSPECT_CMD")
if not INSPECT_CMD:
    # Try common locations
    possible_cmds = [
        "inspect",  # If in PATH
        str(Path.home() / ".local/bin/inspect"),  # Common user install location
        "/usr/local/bin/inspect",  # Homebrew on Intel Mac
        "/opt/homebrew/bin/inspect",  # Homebrew on Apple Silicon
    ]
    for cmd in possible_cmds:
        try:
            result = subprocess.run([cmd, "--version"], capture_output=True, timeout=2)
            if result.returncode == 0:
                INSPECT_CMD = cmd
                print(f"Found inspect at: {cmd}")
                break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    
    if not INSPECT_CMD:
        print("WARNING: inspect command not found. Set INSPECT_CMD environment variable.")
        print("Install inspect_ai: pip install inspect-ai")
        INSPECT_CMD = "inspect"  # Fallback

INSPECT_BASE_CMD: List[str] = [INSPECT_CMD]

RUNS_REGISTRY: Dict[str, Dict[str, Any]] = {}
REGISTRY_LOCK = threading.Lock()
MAX_LOG_LINES = 200


def _now() -> float:
    return time.time()


def _iso_timestamp(timestamp: Optional[float]) -> Optional[str]:
    if not timestamp:
        return None
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return dt.isoformat()


def _format_duration(seconds: Optional[float]) -> Optional[str]:
    if seconds is None:
        return None
    total = int(max(0, seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    parts: List[str] = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if not parts:
        parts.append(f"{secs}s")
        return " ".join(parts)
    if secs and hours < 2:
        parts.append(f"{secs}s")
    return " ".join(parts)


def _progress_ratio(completed: int, total: Optional[int]) -> Optional[float]:
    if not total:
        return None
    ratio = completed / total
    if ratio < 0:
        return 0.0
    if ratio > 1:
        return 1.0
    return ratio


def _resolve_task_name(task_name: str) -> str:
    resolved = TASK_ALIASES.get(task_name, task_name)
    if resolved not in TASK_FILES:
        raise ValueError(f"Unsupported task: {task_name}")
    return resolved


def _estimate_remaining(job: Dict[str, Any]) -> Optional[float]:
    completed = job.get("samples_completed", 0)
    total = job.get("total_samples")
    if not completed:
        return None
    if not total:
        return None
    started_at = job.get("started_at")
    if not started_at:
        return None
    end_point = _now()
    finished_at = job.get("finished_at")
    if finished_at:
        end_point = finished_at
    elapsed = end_point - started_at
    if elapsed <= 0:
        return None
    remaining = total - completed
    if remaining <= 0:
        return 0.0
    average = elapsed / completed
    return remaining * average


def _snapshot_job(job: Dict[str, Any]) -> Dict[str, Any]:
    completed = job.get("samples_completed", 0)
    total_samples = job.get("total_samples")
    ratio = _progress_ratio(completed, total_samples)
    progress_percent = None
    if ratio is not None:
        progress_percent = round(ratio * 100, 2)

    started_at = job.get("started_at")
    end_point = _now()
    finished_at = job.get("finished_at")
    if finished_at:
        end_point = finished_at
    elapsed = None
    if started_at:
        elapsed = end_point - started_at

    eta_seconds = _estimate_remaining(job)

    return {
        "job_id": job.get("job_id"),
        "model": job.get("model"),
        "display_name": job.get("display_name"),
        "provider": job.get("provider"),
        "status": job.get("status"),
        "samples_completed": completed,
        "total_samples": total_samples,
        "progress_percent": progress_percent,
        "elapsed_seconds": elapsed,
        "elapsed": _format_duration(elapsed),
        "eta_seconds": eta_seconds,
        "eta": _format_duration(eta_seconds),
        "error": job.get("error"),
        "started_at": _iso_timestamp(started_at),
        "finished_at": _iso_timestamp(finished_at),
        "returncode": job.get("returncode"),
    }


def _apply_results(job: Dict[str, Any], payload: Dict[str, Any]) -> None:
    results = payload.get("results")
    if isinstance(results, dict):
        total = results.get("total_samples")
        if total is not None:
            job["total_samples"] = total
        completed = results.get("completed_samples")
        if completed is not None:
            job["samples_completed"] = completed

    progress = payload.get("progress")
    if isinstance(progress, dict):
        total = progress.get("total")
        if total is not None:
            job["total_samples"] = total
        completed = progress.get("completed")
        if completed is not None:
            job["samples_completed"] = completed

    sample = payload.get("sample")
    if isinstance(sample, dict):
        total = sample.get("total")
        if total is not None:
            job["total_samples"] = total
        completed = sample.get("completed")
        if isinstance(completed, int):
            if completed > job.get("samples_completed", 0):
                job["samples_completed"] = completed
        index = sample.get("index")
        if isinstance(index, int):
            candidate = index + 1
            if candidate > job.get("samples_completed", 0):
                job["samples_completed"] = candidate

    event = payload.get("event")
    if event in {"sample_complete", "sample_success", "sample"}:
        completed = payload.get("completed")
        if isinstance(completed, int) and completed > job.get("samples_completed", 0):
            job["samples_completed"] = completed


def _handle_progress(job: Dict[str, Any], raw_line: str) -> None:
    job["last_update"] = _now()
    if "log_tail" not in job:
        job["log_tail"] = deque(maxlen=MAX_LOG_LINES)
    job["log_tail"].append(raw_line.rstrip())
    try:
        payload = json.loads(raw_line)
    except json.JSONDecodeError:
        return
    if not isinstance(payload, dict):
        return
    _apply_results(job, payload)


def _build_command(task_name: str, model: str, options: Dict[str, Any]) -> List[str]:
    resolved_task = _resolve_task_name(task_name)
    task_file = TASK_FILES[resolved_task]
    cmd = INSPECT_BASE_CMD + [
        "eval",
        task_file,
        "--display",
        "log",
        "--log-level",
        "info",
        "--log-level-transcript",
        "info",
        "--log-format",
        "json",
        "--model",
        model,
    ]
    
    print(f"[DEBUG] Building command for model: {model}")
    print(f"[DEBUG] Task: {resolved_task}, File: {task_file}")

    difficulty = options.get("difficulty")
    if difficulty and resolved_task == "telemath" and difficulty != "full":
        cmd.extend(["-T", f"difficulty={difficulty}"])

    limit = options.get("limit")
    if limit:
        cmd.extend(["--limit", str(limit)])

    max_connections = options.get("max_connections")
    if max_connections:
        cmd.extend(["--max-connections", str(max_connections)])

    max_tokens = options.get("max_tokens")
    if max_tokens:
        cmd.extend(["--max-tokens", str(max_tokens)])

    temperature = options.get("temperature")
    if temperature is not None and temperature != "":
        cmd.extend(["--temperature", str(temperature)])

    print(f"[DEBUG] Full command: {' '.join(cmd)}")
    return cmd


def _run_inspect_job(run_id: str, job_id: str, task_name: str, command: List[str]) -> None:
    process: Optional[subprocess.Popen[str]] = None
    try:
        with REGISTRY_LOCK:
            run = RUNS_REGISTRY.get(run_id)
            if not run:
                return
            job = run["models"].get(job_id)
            if not job:
                return
            job["status"] = "running"
            job["started_at"] = _now()

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["BUILDKIT_PROGRESS"] = "plain"
        env["DOCKER_BUILDKIT"] = "1"

        src_dir = Path(__file__).parent.parent / "src"
        process = subprocess.Popen(
            command,
            cwd=str(src_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env,
        )

        with REGISTRY_LOCK:
            run = RUNS_REGISTRY.get(run_id)
            if run:
                job = run["models"].get(job_id)
                if job:
                    job["process"] = process

        stream = process.stdout
        if stream is None:
            return_code = process.wait()
        else:
            for raw_line in iter(stream.readline, ""):
                with REGISTRY_LOCK:
                    run = RUNS_REGISTRY.get(run_id)
                    if not run:
                        continue
                    job = run["models"].get(job_id)
                    if not job:
                        continue
                    if job.get("cancel_requested"):
                        break
                    _handle_progress(job, raw_line)

            stream.close()
            return_code = process.wait()

    except Exception as exc:
        print(f"[ERROR] Job {job_id} failed with exception: {exc}")
        import traceback
        traceback.print_exc()
        with REGISTRY_LOCK:
            run = RUNS_REGISTRY.get(run_id)
            if not run:
                return
            job = run["models"].get(job_id)
            if not job:
                return
            job["status"] = "failed"
            job["error"] = str(exc)
            job["finished_at"] = _now()
            job["process"] = None
        return

    with REGISTRY_LOCK:
        run = RUNS_REGISTRY.get(run_id)
        if not run:
            return
        job = run["models"].get(job_id)
        if not job:
            return
        job["process"] = None
        job["returncode"] = return_code
        job["finished_at"] = _now()
        if job.get("cancel_requested"):
            job["status"] = "cancelled"
            return
        if return_code == 0:
            job["status"] = "complete"
            return
        job["status"] = "failed"
        error_msg = f"Exited with code {return_code}"
        print(f"[ERROR] Job {job_id} failed: {error_msg}")
        print(f"[ERROR] Last log lines:")
        for line in job.get("log_tail", []):
            print(f"  {line}")
        job["error"] = error_msg


def _register_run(run_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    requested_task = payload.get("task", "telemath")
    task = _resolve_task_name(requested_task)

    models_payload = payload.get("models", [])
    if not models_payload:
        raise ValueError("At least one model must be provided")

    options = {
        "difficulty": payload.get("difficulty"),
        "limit": payload.get("limit"),
        "max_connections": payload.get("max_connections"),
        "max_tokens": payload.get("max_tokens"),
        "temperature": payload.get("temperature"),
    }

    run_entry = {
        "run_id": run_id,
        "task": task,
        "created_at": _now(),
        "options": options,
        "models": {},
    }

    with REGISTRY_LOCK:
        RUNS_REGISTRY[run_id] = run_entry

    for model_info in models_payload:
        model_name = model_info.get("model")
        if not model_name:
            continue
        job_id = uuid.uuid4().hex
        display_name = model_info.get("label") or model_name.split("/")[-1]
        provider = model_info.get("provider")
        job = {
            "job_id": job_id,
            "model": model_name,
            "display_name": display_name,
            "provider": provider,
            "status": "queued",
            "samples_completed": 0,
            "total_samples": None,
            "error": None,
            "started_at": None,
            "finished_at": None,
            "process": None,
            "cancel_requested": False,
            "returncode": None,
            "log_tail": deque(maxlen=MAX_LOG_LINES),
        }
        with REGISTRY_LOCK:
            retrieved_run = RUNS_REGISTRY.get(run_id)
            if retrieved_run is None:
                continue
            retrieved_run["models"][job_id] = job
        command = _build_command(task, model_name, options)
        worker = threading.Thread(
            target=_run_inspect_job,
            args=(run_id, job_id, task, command),
            daemon=True,
        )
        job["thread"] = worker
        worker.start()

    return run_entry


@app.route('/api/runs', methods=['POST'])
def create_run():
    payload = request.get_json(silent=True) or {}
    run_id = uuid.uuid4().hex
    try:
        run_entry = _register_run(run_id, payload)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    with REGISTRY_LOCK:
        run = RUNS_REGISTRY.get(run_id)
        if not run:
            return jsonify({'error': 'Run registration failed'}), 500
        models_snapshot = [
            {
                'job_id': job['job_id'],
                'model': job['model'],
                'display_name': job.get('display_name'),
                'provider': job.get('provider'),
            }
            for job in run['models'].values()
        ]

    response = {
        'run_id': run_id,
        'task': run_entry['task'],
        'created_at': _iso_timestamp(run_entry['created_at']),
        'models': models_snapshot,
        'options': run_entry['options'],
    }
    return jsonify(response)


@app.route('/api/runs/<run_id>/status')
def run_status(run_id: str):
    with REGISTRY_LOCK:
        run = RUNS_REGISTRY.get(run_id)
        if not run:
            return jsonify({'error': 'Run not found'}), 404
        snapshots = [_snapshot_job(job) for job in run['models'].values()]
        total_jobs = len(snapshots)
        completed = sum(1 for item in snapshots if item['status'] == 'complete')
        failed = sum(1 for item in snapshots if item['status'] == 'failed')
        cancelled = sum(1 for item in snapshots if item['status'] == 'cancelled')
        running_jobs = sum(1 for item in snapshots if item['status'] == 'running')
        queued_jobs = sum(1 for item in snapshots if item['status'] == 'queued')
        last_update_candidates = [run['created_at']]
        last_update_candidates.extend(
            job.get('last_update') for job in run['models'].values() if job.get('last_update')
        )

    overall_status = 'running'
    if total_jobs == 0:
        overall_status = 'empty'
    if cancelled == total_jobs and total_jobs:
        overall_status = 'cancelled'
    if failed and failed + completed + cancelled == total_jobs:
        overall_status = 'failed'
    if completed == total_jobs and total_jobs:
        overall_status = 'complete'
    if running_jobs == 0 and queued_jobs == total_jobs and total_jobs:
        overall_status = 'queued'

    last_update = max(filter(None, last_update_candidates)) if last_update_candidates else None

    response = {
        'run_id': run_id,
        'task': run['task'],
        'created_at': _iso_timestamp(run['created_at']),
        'updated_at': _iso_timestamp(last_update),
        'overall': {
            'total': total_jobs,
            'complete': completed,
            'failed': failed,
            'cancelled': cancelled,
            'running': running_jobs,
            'queued': queued_jobs,
            'status': overall_status,
        },
        'models': snapshots,
    }
    return jsonify(response)


@app.route('/api/runs/<run_id>/cancel', methods=['POST'])
def cancel_run(run_id: str):
    with REGISTRY_LOCK:
        run = RUNS_REGISTRY.get(run_id)
        if not run:
            return jsonify({'error': 'Run not found'}), 404
        for job in run['models'].values():
            job['cancel_requested'] = True
            job['status'] = 'cancelling'
            process = job.get('process')
            if not process:
                continue
            if process.poll() is not None:
                continue
            try:
                process.terminate()
            except Exception:
                process.kill()

    return jsonify({'run_id': run_id, 'status': 'cancelling'})

# Function to read eval logs using inspect
def read_eval_log(log_path):
    try:
        # Use inspect log command to read the eval file
        result = subprocess.run(
            INSPECT_BASE_CMD + ['log', 'info', log_path, '--json'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
        return None
    except Exception as e:
        print(f"Error reading log: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run', methods=['POST'])
def run_evaluation():
    data = request.json
    
    try:
        task = _resolve_task_name(data.get('task', 'telemath'))
    except ValueError as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400

    task_file = TASK_FILES[task]

    # Build the command
    cmd = INSPECT_BASE_CMD + ['eval', task_file]
    
    # Add model
    if data.get('model'):
        cmd.extend(['--model', data['model']])
    
    # Add difficulty as task parameter (only applies to telemath)
    if task == 'telemath' and data.get('difficulty') and data['difficulty'] != 'full':
        cmd.extend(['-T', f'difficulty={data["difficulty"]}'])
    
    # Add max connections
    if data.get('max_connections'):
        cmd.extend(['--max-connections', str(data['max_connections'])])
    
    # Add max tokens
    if data.get('max_tokens'):
        cmd.extend(['--max-tokens', str(data['max_tokens'])])
    
    # Add limit
    if data.get('limit'):
        cmd.extend(['--limit', str(data['limit'])])
    
    # Add temperature
    if data.get('temperature'):
        cmd.extend(['--temperature', str(data['temperature'])])
    
    # Change to src directory
    src_dir = Path(__file__).parent.parent / 'src'
    
    try:
        # Run the command
        process = subprocess.Popen(
            cmd,
            cwd=str(src_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Read output
        output = []
        for line in process.stdout:
            output.append(line)
        
        process.wait()
        
        return jsonify({
            'success': True,
            'output': ''.join(output),
            'returncode': process.returncode
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/stream')
def stream():
    # Capture request args outside the generator to avoid context issues
    data = request.args.to_dict()
    
    def generate():
        # Get task type (default to telemath for backward compatibility)
        try:
            task = _resolve_task_name(data.get('task', 'telemath'))
        except ValueError as exc:
            yield f"data: [ERROR] {str(exc)}\n\n"
            return
        task_file = TASK_FILES[task]
        
        # Build the command
        cmd = INSPECT_BASE_CMD + [
            'eval', task_file,
            '--display', 'log',  # Use log display for better streaming
            '--log-level', 'info',  # Show info level logs
            '--log-level-transcript', 'info',  # Include info in transcript
            '--log-format', 'json'  # Output logs in JSON format
        ]
        
        if data.get('model'):
            cmd.extend(['--model', data['model']])
        
        # Only add difficulty for telemath
        if task == 'telemath' and data.get('difficulty') and data['difficulty'] != 'full':
            cmd.extend(['-T', f'difficulty={data["difficulty"]}'])
        
        if data.get('max_connections'):
            cmd.extend(['--max-connections', data['max_connections']])
        
        if data.get('max_tokens'):
            cmd.extend(['--max-tokens', data['max_tokens']])
        
        if data.get('limit'):
            cmd.extend(['--limit', data['limit']])
        
        if data.get('temperature'):
            cmd.extend(['--temperature', data['temperature']])
        
        src_dir = Path(__file__).parent.parent / 'src'
        
        def enqueue_output(out, queue):
            for line in iter(out.readline, ''):
                queue.put(line)
            out.close()
        
        try:
            # Set environment to disable buffering
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            env['BUILDKIT_PROGRESS'] = 'plain'  # Show plain Docker build output
            env['DOCKER_BUILDKIT'] = '1'
            
            process = subprocess.Popen(
                cmd,
                cwd=str(src_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                universal_newlines=True,
                bufsize=1
            )
            
            # Use a queue and thread to read output asynchronously
            q = Queue()
            t = Thread(target=enqueue_output, args=(process.stdout, q))
            t.daemon = True
            t.start()
            
            # Yield output as it arrives
            while True:
                try:
                    line = q.get(timeout=0.1)
                    yield f"data: {line}\n\n"
                except Empty:
                    # Check if process has finished
                    if process.poll() is not None:
                        # Get any remaining lines
                        while not q.empty():
                            try:
                                line = q.get_nowait()
                                yield f"data: {line}\n\n"
                            except Empty:
                                break
                        break
            
            return_code = process.wait()
            yield f"data: \n[DONE] Exit code: {return_code}\n\n"
        
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/logs')
def list_logs():
    """List all available JSON log files"""
    src_logs_dir = Path(__file__).parent.parent / 'src' / 'logs'
    root_logs_dir = Path(__file__).parent.parent / 'logs'
    
    logs = []
    
    # Check both log directories - ONLY JSON files
    for logs_dir in [src_logs_dir, root_logs_dir]:
        if logs_dir.exists():
            for log_file in sorted(logs_dir.glob('*.json'), 
                                   key=lambda x: x.stat().st_mtime, reverse=True):
                logs.append({
                    'name': log_file.name,
                    'path': str(log_file),
                    'size': log_file.stat().st_size,
                    'modified': log_file.stat().st_mtime,
                    'type': 'json'
                })
    
    return jsonify(logs)

@app.route('/api/logs/<path:log_name>')
def get_log(log_name):
    """Get detailed information about a specific JSON log file"""
    src_logs_dir = Path(__file__).parent.parent / 'src' / 'logs'
    root_logs_dir = Path(__file__).parent.parent / 'logs'
    
    # Try to find the log file
    log_path = None
    for logs_dir in [src_logs_dir, root_logs_dir]:
        candidate = logs_dir / log_name
        if candidate.exists():
            log_path = candidate
            break
    
    if not log_path:
        return jsonify({'error': 'Log file not found'}), 404
    
    # Only accept JSON files
    if log_path.suffix != '.json':
        return jsonify({'error': 'Only JSON log files are supported. Please run evaluations with --log-format=json'}), 400
    
    try:
        with open(log_path, 'r') as f:
            log_data = json.load(f)
        return jsonify(log_data)
    except Exception as e:
        return jsonify({'error': f'Failed to read JSON file: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001, host='0.0.0.0')

