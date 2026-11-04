"""Evidence-based security recommendations.

Each recommendation is produced by a rule that looks at the *actual*
detections. A rule only fires when its evidence threshold is met, and every
recommendation reports the numbers that triggered it. Nothing is random.
"""
from __future__ import annotations

import pandas as pd

PRIORITY_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}


def _top(df: pd.DataFrame, col: str = "user_id", n: int = 3) -> str:
    if df.empty:
        return "-"
    vc = df[col].value_counts().head(n)
    return ", ".join(f"{k} ({v})" for k, v in vc.items())


def _rec(priority, title, finding, action, why, affected, count) -> dict:
    return {
        "priority": priority,
        "title": title,
        "finding": finding,
        "action": action,
        "why": why,
        "affected": affected,
        "events": int(count),
    }


def generate_recommendations(scored: pd.DataFrame, overall: dict | None = None) -> pd.DataFrame:
    """Return recommendations (sorted by priority) linked to detected findings."""
    recs: list[dict] = []
    if scored.empty:
        return pd.DataFrame(recs)

    flagged = scored[scored["classification"].astype(str) != "Normal"]
    high = scored[scored["classification"].astype(str) == "High Risk"]
    n = len(scored)

    # 1. Credential attacks -> MFA + review logins + lockout
    brute = scored[scored["failed_logins"] >= 5]
    if len(brute) >= 3:
        pr = "Critical" if (brute["failed_logins"] >= 15).sum() >= 3 else "High"
        recs.append(_rec(
            pr, "Enable Multi-Factor Authentication and account lockout",
            f"{len(brute)} records show >=5 failed logins (max {int(brute['failed_logins'].max())} in one record).",
            "Enforce MFA for the affected accounts, add progressive lockout / rate limiting, and review the failed-login log.",
            "Repeated failed logins are the typical footprint of password guessing or credential stuffing; MFA blocks "
            "a correct guessed password from being enough.",
            _top(brute), len(brute)))

    # 2. Suspicious logins from new locations
    newloc = flagged[flagged["new_location"] == 1]
    if len(newloc) >= 2:
        recs.append(_rec(
            "High", "Review suspicious logins from new locations",
            f"{len(newloc)} flagged records come from a never-before-seen location.",
            "Verify with the account owners, force re-authentication, and apply conditional-access (geo/device) policies.",
            "An unfamiliar location combined with other unusual behaviour may indicate a stolen credential.",
            _top(newloc), len(newloc)))

    # 3. Port scanning / recon
    scan = scored[scored["z_unique_ports"] > 6]
    if len(scan) >= 3:
        recs.append(_rec(
            "High", "Restrict unnecessary network access and review firewall rules",
            f"{len(scan)} records contact an unusually large number of distinct ports "
            f"(max {int(scan['unique_ports'].max())}).",
            "Block or rate-limit the sources, close unused ports, apply default-deny segmentation and review IDS/firewall rules.",
            "Touching hundreds of ports in a short session is typical reconnaissance (port scanning) that precedes exploitation.",
            _top(scan, "source_ip"), len(scan)))

    # 4. Data exfiltration
    exfil = scored[scored["z_log_bytes_out"] > 6]
    if len(exfil) >= 2:
        off = int((exfil["off_hours"] == 1).sum())
        recs.append(_rec(
            "Critical" if off >= 2 else "High",
            "Investigate unusual outbound data transfers",
            f"{len(exfil)} records have abnormally large outbound volume "
            f"(max {exfil['bytes_out_mb'].max():,.0f} MB, {off} outside business hours).",
            "Inspect destinations and content, enable DLP / egress monitoring, and temporarily restrict transfer rights if justified.",
            "Large outbound transfers, especially off-hours, can indicate data theft or an insider issue.",
            _top(exfil), len(exfil)))

    # 5. Traffic surges
    surge = scored[(scored["z_connections"] > 6) & (scored["z_unique_ports"] <= 6)]
    if len(surge) >= 3:
        recs.append(_rec(
            "Medium", "Apply rate limiting and traffic monitoring",
            f"{len(surge)} records show connection counts far above the user's normal baseline "
            f"(max {int(surge['connections'].max())}).",
            "Enable rate limiting / DDoS protection, check for compromised hosts generating beacon-like traffic, review proxy logs.",
            "A sudden surge of connections can signal DDoS, malware beaconing or a misconfigured service.",
            _top(surge), len(surge)))

    # 6. Off-hours access
    off_flag = flagged[flagged["off_hours"] == 1]
    if len(off_flag) >= 5:
        recs.append(_rec(
            "Medium", "Review off-hours access policies",
            f"{len(off_flag)} flagged records occur between 22:00 and 05:59.",
            "Confirm legitimate need, restrict after-hours access for non-essential roles, and alert on after-hours admin logins.",
            "Attackers often operate outside business hours when monitoring is thin.",
            _top(off_flag), len(off_flag)))

    # 7. Containment for the most serious cases
    if len(high) >= 5:
        worst = high.sort_values("event_risk", ascending=False) if "event_risk" in high else high
        recs.append(_rec(
            "Critical", "Consider isolating the most affected endpoints (authorized personnel only)",
            f"{len(high)} records are classified High Risk; most affected: {_top(worst, 'user_id', 3)}.",
            "After confirmation by the security team and with proper authorization, isolate the endpoint(s), "
            "preserve logs/evidence, and start incident response.",
            "Containment limits lateral movement - but only when the evidence justifies it, because isolation "
            "interrupts legitimate work and anomalies are not proof of compromise.",
            _top(worst), len(high)))

    # 8. Overall posture
    if overall:
        lvl = overall.get("level")
        if lvl in ("High", "Critical"):
            recs.append(_rec(
                "High", "Raise monitoring and run an incident-readiness check",
                f"Overall risk score is {overall['score']} ({lvl}).",
                "Increase log retention and alerting, confirm on-call procedures, and prioritise the actions above.",
                "A high aggregate score means several risk factors are elevated at the same time.",
                "organisation-wide", n))

    if not recs:
        recs.append(_rec(
            "Low", "Maintain current security hygiene",
            "No finding reached a recommendation threshold in this dataset.",
            "Keep MFA, patching and logging in place and re-run the analysis regularly.",
            "A quiet result is reassuring but does not prove the absence of attacks - detection has blind spots.",
            "-", 0))

    out = pd.DataFrame(recs)
    out["_o"] = out["priority"].map(PRIORITY_ORDER)
    return out.sort_values(["_o", "events"], ascending=[True, False]).drop(columns="_o").reset_index(drop=True)
