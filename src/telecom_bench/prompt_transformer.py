"""Transform telecom prompts to Markdown-KV format.

This module transforms pipe-delimited and CSV telecom evaluation prompts into
Markdown-KV format for optimal LLM comprehension. Markdown-KV format achieves
~60% accuracy vs ~41-44% for pipe-delimited/CSV in LLM table understanding benchmarks.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class ColumnMapping:
    """Maps original column names to abbreviated keys."""

    PATTERNS: Dict[str, str] = field(default_factory=lambda: {
        r"5G KPI PCell RF Serving SS-RSRP \[dBm\]": "ss_rsrp_dbm",
        r"5G KPI PCell RF Serving SS-SINR \[dB\]": "ss_sinr_db",
        r"5G KPI PCell RF Serving PCI": "serving_pci",
        r"5G KPI PCell Layer2 MAC DL Throughput \[Mbps\]": "dl_tput_mbps",
        r"5G KPI PCell Layer1 DL RB Num.*": "dl_rb_num",
        r"Measurement.*Top (\d+) PCI": r"neighbor_pci_\1",
        r"Measurement.*Top (\d+).*BRSRP \[dBm\]": r"neighbor_rsrp_\1",
        r"GPS Speed.*": "speed_kmh",
        r"Longitude": "lon",
        r"Latitude": "lat",
        r"Timestamp": "timestamp",
        r"Mechanical Azimuth": "azimuth_mech",
        r"Mechanical Downtilt": "downtilt_mech",
        r"Digital Tilt": "downtilt_digital",
        r"Digital Azimuth": "azimuth_digital",
        r"Max Transmit Power": "max_tx_power",
        r"TxRx Mode": "txrx_mode",
        r"Beam Scenario": "beam_scenario",
        r"Antenna Model": "antenna_model",
        r"gNodeB ID": "gnodeb_id",
        r"Cell ID": "cell_id",
        r"Height": "height_m",
        r"PCI": "pci",
    })

    def transform(self, col_name: str) -> str:
        """Return abbreviated key for column name."""
        for pattern, replacement in self.PATTERNS.items():
            match = re.match(pattern, col_name, re.IGNORECASE)
            if match:
                # Handle captured groups (e.g., neighbor_pci_1)
                if '\\1' in replacement and match.groups():
                    return replacement.replace('\\1', match.group(1))
                return replacement
        # Fallback: convert to snake_case
        return col_name.lower().replace(" ", "_").replace("[", "").replace("]", "")


def parse_pipe_delimited(text: str) -> Tuple[List[str], List[List[str]]]:
    """Parse pipe-delimited table into headers and rows.

    Args:
        text: Pipe-delimited table text

    Returns:
        Tuple of (headers, rows)
    """
    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip() and not ln.strip().startswith("---")]
    if not lines:
        return [], []

    # First line is headers
    headers = [h.strip() for h in lines[0].split("|") if h.strip()]

    # Remaining lines are data rows
    rows = []
    for ln in lines[1:]:
        # Skip separator lines
        if re.match(r'^[\s\|\-]+$', ln):
            continue
        row = [cell.strip() for cell in ln.split("|") if cell.strip() or cell == ""]
        if row:  # Only add non-empty rows
            rows.append(row)

    return headers, rows


def aggregate_neighbors(record: Dict[str, str]) -> Dict[str, any]:
    """Collapse neighbor_pci_N and neighbor_rsrp_N into lists.

    Args:
        record: Dictionary with potentially multiple neighbor_pci_X and neighbor_rsrp_X keys

    Returns:
        Modified record with neighbor_pcis and neighbor_rsrp_dbm lists
    """
    pcis = []
    rsrps = []
    keys_to_remove = []

    # Look for up to 10 neighbors
    for i in range(1, 11):
        pci_key = f"neighbor_pci_{i}"
        rsrp_key = f"neighbor_rsrp_{i}"

        if pci_key in record and record[pci_key] not in ("-", "", None, "None"):
            pcis.append(record[pci_key])
            keys_to_remove.append(pci_key)
        if rsrp_key in record and record[rsrp_key] not in ("-", "", None, "None"):
            rsrps.append(record[rsrp_key])
            keys_to_remove.append(rsrp_key)

    # Remove individual neighbor keys
    for key in keys_to_remove:
        del record[key]

    # Add aggregated lists
    if pcis:
        record["neighbor_pcis"] = pcis
    if rsrps:
        record["neighbor_rsrp_dbm"] = rsrps

    return record


def record_to_markdown_kv(record: Dict[str, any], record_id: str) -> str:
    """Convert single record dict to Markdown-KV block.

    Args:
        record: Dictionary of key-value pairs
        record_id: Identifier for this record

    Returns:
        Markdown-formatted string with KV pairs
    """
    lines = [f"## {record_id}", "```"]

    for key, value in record.items():
        # Skip null/empty values
        if value in ("-", "", None, "None"):
            continue

        # Format lists with brackets
        if isinstance(value, list):
            value = f"[{', '.join(str(v) for v in value)}]"

        lines.append(f"{key}: {value}")

    lines.append("```")
    return "\n".join(lines)


def transform_drive_test_table(table_text: str) -> str:
    """Transform drive test pipe-delimited table to Markdown-KV.

    Args:
        table_text: Pipe-delimited table string

    Returns:
        Markdown-KV formatted string
    """
    mapper = ColumnMapping()
    headers, rows = parse_pipe_delimited(table_text)

    if not headers or not rows:
        return ""

    # Transform column names
    transformed_headers = [mapper.transform(h) for h in headers]

    output_lines = ["# Drive Test Data", ""]

    for i, row in enumerate(rows, 1):
        # Pad row if it's shorter than headers
        while len(row) < len(transformed_headers):
            row.append("")

        record = dict(zip(transformed_headers, row))
        record = aggregate_neighbors(record)

        # Use timestamp for record ID if available
        timestamp = record.get("timestamp", f"Record {i}")
        record_id = f"Record {i} ({timestamp})" if "timestamp" in record else f"Record {i}"

        output_lines.append(record_to_markdown_kv(record, record_id))
        output_lines.append("")

    return "\n".join(output_lines)


def transform_engineering_params(table_text: str) -> str:
    """Transform engineering parameters table to Markdown-KV grouped by PCI.

    Args:
        table_text: Pipe-delimited table string

    Returns:
        Markdown-KV formatted string
    """
    mapper = ColumnMapping()
    headers, rows = parse_pipe_delimited(table_text)

    if not headers or not rows:
        return ""

    # Transform column names
    transformed_headers = [mapper.transform(h) for h in headers]

    output_lines = ["# Engineering Parameters", ""]

    for row in rows:
        # Pad row if it's shorter than headers
        while len(row) < len(transformed_headers):
            row.append("")

        record = dict(zip(transformed_headers, row))
        pci = record.get("pci", "Unknown")

        output_lines.append(record_to_markdown_kv(record, f"Cell PCI={pci}"))
        output_lines.append("")

    return "\n".join(output_lines)


def extract_domain_rules(text: str) -> str:
    """Extract and reformat domain rules section.

    Args:
        text: Original prompt text with domain rules

    Returns:
        Formatted domain rules in Markdown
    """
    rules_section = []
    rules_section.append("# Domain Rules\n")

    # Look for "Given:" section
    given_match = re.search(r'Given:(.*?)(?=User plane|Engineering|$)', text, re.IGNORECASE | re.DOTALL)

    if given_match:
        given_text = given_match.group(1).strip()

        # Parse bullet points
        bullet_points = re.findall(r'[-•]\s*(.+?)(?=\n[-•]|\n\n|$)', given_text, re.DOTALL)

        if bullet_points:
            # Group rules by topic
            downtilt_rules = []
            beam_rules = []
            other_rules = []

            for point in bullet_points:
                point = point.strip()
                if "downtilt" in point.lower() or "255" in point:
                    downtilt_rules.append(point)
                elif "beam scenario" in point.lower() or "beamwidth" in point.lower():
                    beam_rules.append(point)
                else:
                    other_rules.append(point)

            # Format downtilt rules
            if downtilt_rules:
                rules_section.append("## Downtilt Interpretation")
                for rule in downtilt_rules:
                    if "255" in rule:
                        rules_section.append("- Value 255 → 6° (default)")
                        rules_section.append("- Other values → actual degrees")
                    else:
                        rules_section.append(f"- {rule}")
                rules_section.append("")

            # Format beam scenario rules - keep original format
            if beam_rules:
                rules_section.append("## Beam Scenario and Vertical Beamwidth Relationships")
                for rule in beam_rules:
                    rules_section.append(f"- {rule}")
                rules_section.append("")

            # Format other rules
            if other_rules:
                rules_section.append("## Additional Rules")
                for rule in other_rules:
                    rules_section.append(f"- {rule}")
                rules_section.append("")

    return "\n".join(rules_section)


def transform_prompt(prompt: str) -> str:
    """Full transformation pipeline for telecom prompt.

    Transforms pipe-delimited tables to Markdown-KV format with:
    1. Domain rules at the top
    2. Drive test data (time-series)
    3. Engineering parameters (reference data)
    4. Data relationships declaration

    Args:
        prompt: Original prompt with pipe-delimited tables

    Returns:
        Transformed prompt in Markdown-KV format
    """
    sections = []

    # 1. Extract and transform domain rules
    domain_rules = extract_domain_rules(prompt)
    if domain_rules:
        sections.append(domain_rules)

    # 2. Find and transform drive test data
    # Look for patterns like "User plane drive test data" or "drive test data"
    drive_test_match = re.search(
        r'(?:User plane\s+)?(?:drive test|test)\s+data.*?[:：]\s*\n+(.*?)(?=\n+(?:Engineering|Eng[a-z]*\s+parameters|$))',
        prompt,
        re.IGNORECASE | re.DOTALL
    )

    if drive_test_match:
        drive_test_text = drive_test_match.group(1).strip()
        transformed_drive = transform_drive_test_table(drive_test_text)
        if transformed_drive:
            sections.append(transformed_drive)

    # 3. Find and transform engineering parameters
    eng_params_match = re.search(
        r'Eng(?:ineering)?\s+parameters.*?[:：]\s*\n+(.*?)(?=\n\n[A-Z]|$)',
        prompt,
        re.IGNORECASE | re.DOTALL
    )

    if eng_params_match:
        eng_params_text = eng_params_match.group(1).strip()
        transformed_eng = transform_engineering_params(eng_params_text)
        if transformed_eng:
            sections.append(transformed_eng)

    # 4. Add relationship declaration
    sections.append("""# Data Relationships
- Drive test `serving_pci` and `neighbor_pcis` link to Engineering Parameters via `PCI`
""")

    return "\n".join(sections)


def validate_transformation(original: str, transformed: str) -> Dict[str, bool]:
    """Validate that transformation meets quality criteria.

    Args:
        original: Original prompt text
        transformed: Transformed prompt text

    Returns:
        Dictionary of validation checks
    """
    # Check for null/empty value patterns (key: - or key: null)
    # but exclude markdown list items (- at start of line)
    has_null_values = bool(
        re.search(r':\s*-\s*\n', transformed) or
        re.search(r':\s*null\s*\n', transformed) or
        re.search(r':\s*None\s*\n', transformed)
    )

    # Check if pipe delimiters are still present in table format
    # Allow pipes in list values like [val1, val2]
    has_pipe_tables = bool(re.search(r'\|.*\|.*\|', transformed))

    checks = {
        "has_domain_rules": "# Domain Rules" in transformed,
        "has_drive_test": "# Drive Test Data" in transformed,
        "has_engineering_params": "# Engineering Parameters" in transformed,
        "has_relationships": "# Data Relationships" in transformed,
        "no_pipe_delimiters": not has_pipe_tables,
        "has_code_blocks": "```" in transformed,
        "no_null_values": not has_null_values,
    }

    return checks
