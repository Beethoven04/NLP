#!/usr/bin/env python3
import csv
import json
import random
from datetime import datetime, timedelta
from pathlib import Path


random.seed(42)

OUT_JSON = Path("/Users/beethoven/NLP/devoir3/dataset/custom_snort_rag_dataset_500.json")
OUT_CSV = Path("/Users/beethoven/NLP/devoir3/dataset/custom_snort_rag_dataset_500.csv")


ATTACK_PROFILES = [
    {"attack_name": "Port Scanning", "attack_family": "Reconnaissance", "protocol": "tcp", "ports": ["21", "22", "23", "25", "53", "80", "110", "139", "143", "443", "445", "3389"], "classtype": "attempted-recon", "severity": "medium"},
    {"attack_name": "SYN Flood", "attack_family": "DoS", "protocol": "tcp", "ports": ["80", "443"], "classtype": "attempted-dos", "severity": "high"},
    {"attack_name": "DDoS Traffic Burst", "attack_family": "DDoS", "protocol": "udp", "ports": ["53", "123", "1900"], "classtype": "attempted-dos", "severity": "critical"},
    {"attack_name": "SQL Injection", "attack_family": "Web Attack", "protocol": "http", "ports": ["80", "443"], "classtype": "web-application-attack", "severity": "high"},
    {"attack_name": "Cross-Site Scripting (XSS)", "attack_family": "Web Attack", "protocol": "http", "ports": ["80", "443"], "classtype": "web-application-attack", "severity": "medium"},
    {"attack_name": "DNS Tunneling", "attack_family": "C2/Exfiltration", "protocol": "dns", "ports": ["53"], "classtype": "bad-unknown", "severity": "high"},
    {"attack_name": "SSH Bruteforce", "attack_family": "Credential Access", "protocol": "tcp", "ports": ["22"], "classtype": "attempted-admin", "severity": "high"},
    {"attack_name": "Malware Beaconing", "attack_family": "Malware Traffic", "protocol": "http", "ports": ["80", "443", "8080"], "classtype": "trojan-activity", "severity": "high"},
    {"attack_name": "C2 Keepalive Channel", "attack_family": "Command and Control", "protocol": "tcp", "ports": ["443", "8080", "8443"], "classtype": "trojan-activity", "severity": "critical"},
    {"attack_name": "ICMP Abuse Sweep", "attack_family": "Reconnaissance", "protocol": "icmp", "ports": ["any"], "classtype": "attempted-recon", "severity": "medium"},
    {"attack_name": "HTTP Exploit Attempt", "attack_family": "Initial Access", "protocol": "http", "ports": ["80", "443"], "classtype": "web-application-attack", "severity": "high"},
    {"attack_name": "FTP Abuse", "attack_family": "Lateral Movement", "protocol": "tcp", "ports": ["21"], "classtype": "policy-violation", "severity": "medium"},
    {"attack_name": "SMTP Abuse / Phishing Relay", "attack_family": "Collection/Delivery", "protocol": "tcp", "ports": ["25", "587"], "classtype": "policy-violation", "severity": "medium"},
    {"attack_name": "Network Service Recon", "attack_family": "Reconnaissance", "protocol": "tcp", "ports": ["135", "139", "445", "3389"], "classtype": "attempted-recon", "severity": "medium"},
    {"attack_name": "Data Exfiltration over HTTPS", "attack_family": "Exfiltration", "protocol": "http", "ports": ["443"], "classtype": "data-theft", "severity": "critical"},
]

NETWORK_ZONES = ["DMZ web tier", "east-west server VLAN", "hybrid cloud subnet", "corporate user LAN", "OT gateway segment", "remote access zone"]
ASSETS = ["payment-api-01", "hr-portal-02", "vpn-gateway-01", "ad-dc-02", "db-cluster-01", "mail-edge-01", "k8s-ingress-01"]
USER_AGENTS = ["curl/8.4.0", "python-requests/2.32", "Mozilla/5.0", "Go-http-client/2.0", "Wget/1.21"]
SQL_PATTERNS = ["' OR 1=1 --", "UNION SELECT user,password FROM users", "'; DROP TABLE sessions; --", "SLEEP(5)", "information_schema.tables"]
XSS_PATTERNS = ["<script>alert('x')</script>", "onerror=alert(1)", "<img src=x onload=fetch('/cookie')>", "javascript:alert(document.domain)"]
MALWARE_PATHS = ["/api/v1/ping", "/gate/checkin", "/sync/telemetry", "/cdn/style.php", "/assets/update"]
MAIL_SUBJECTS = ["Urgent invoice review", "Password expiry notice", "Confidential payroll update", "Security policy acknowledgment"]


def rand_ip(private=False):
    if private:
        base = random.choice([(10, random.randint(0, 255), random.randint(0, 255)),
                              (172, random.randint(16, 31), random.randint(0, 255)),
                              (192, 168, random.randint(0, 255))])
        return f"{base[0]}.{base[1]}.{base[2]}.{random.randint(2, 254)}"
    return f"{random.randint(11, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"


def ts(offset):
    t0 = datetime(2026, 4, 1, 0, 0, 0) + timedelta(seconds=offset)
    return t0.strftime("%Y-%m-%dT%H:%M:%SZ")


def build_network_log(profile, src_ip, dst_ip, src_port, dst_port, idx):
    proto = profile["protocol"]
    stamp = ts(idx * 57)
    if proto == "icmp":
        return f"{stamp} sensor=ids-east src={src_ip} dst={dst_ip} proto=ICMP type=8 code=0 packets={random.randint(20, 300)} size={random.randint(64, 512)} ttl={random.randint(40, 64)}"
    if proto == "dns":
        token = "".join(random.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(random.randint(35, 52)))
        return f'{stamp} sensor=resolver-02 src={src_ip}:{src_port} dst={dst_ip}:53 proto=UDP query="{token}.corp-check.net" qtype=TXT rcode=NOERROR bytes_out={random.randint(300, 1200)} bytes_in={random.randint(80, 200)}'
    if proto == "http":
        ua = random.choice(USER_AGENTS)
        uri = random.choice(MALWARE_PATHS + ["/login", "/search", "/admin", "/api/export", "/wp-admin/admin-ajax.php"])
        method = random.choice(["GET", "POST"])
        status = random.choice([200, 200, 302, 403, 500])
        return f'{stamp} sensor=proxy-west src={src_ip}:{src_port} dst={dst_ip}:{dst_port} proto=TCP app=HTTP method={method} uri="{uri}" status={status} ua="{ua}" bytes={random.randint(300, 8000)}'
    return f"{stamp} sensor=core-ids src={src_ip}:{src_port} dst={dst_ip}:{dst_port} proto={proto.upper()} flags={random.choice(['S','PA','A','F'])} packets={random.randint(4, 800)} bytes={random.randint(200, 120000)}"


def indicators_for(profile):
    name = profile["attack_name"]
    if "SQL" in name:
        return [random.choice(SQL_PATTERNS), "suspicious query string", "unexpected DB error burst", "repeated POST to login/search"]
    if "XSS" in name:
        return [random.choice(XSS_PATTERNS), "encoded script payload", "reflected input in response body"]
    if "DNS Tunneling" in name:
        return ["high-entropy subdomain labels", "long TXT queries", "steady low-volume DNS egress", "beacon interval consistency"]
    if "SSH Bruteforce" in name:
        return ["multiple failed SSH auth", "username spray pattern", "rapid reconnect from single source"]
    if "Exfiltration" in name:
        return ["large outbound transfer after archive creation", "rare external endpoint", "TLS session duration anomaly", "off-hours egress spike"]
    if "SYN Flood" in name or "DDoS" in name:
        return ["packet rate spike", "asymmetric SYN/ACK ratio", "source diversity anomaly", "service saturation symptoms"]
    if "C2" in name or "Beaconing" in name:
        return ["periodic callback interval", "low-and-slow encrypted channel", "JA3/UA mismatch", "domain age anomaly"]
    if "SMTP" in name:
        return ["high outbound SMTP rate", "spoofed sender patterns", "credential harvesting lure language"]
    if "FTP" in name:
        return ["unexpected FTP control commands", "anonymous login attempts", "bulk file pull sequence"]
    if "ICMP" in name:
        return ["ICMP echo sweep cadence", "non-standard payload size", "subnet-wide probing sequence"]
    return ["multi-port probing", "service fingerprint attempts", "short inter-packet interval"]


def attack_descriptions(profile):
    a = profile["attack_name"]
    short = {
        "Port Scanning": "External host mapped exposed services across multiple ports before follow-on access attempts.",
        "SYN Flood": "Burst of half-open TCP handshakes attempted to exhaust listener backlog on internet-facing service.",
        "DDoS Traffic Burst": "Distributed reflected/amplified traffic targeted a critical service to degrade availability.",
        "SQL Injection": "Input fields were abused with SQL operators to bypass logic and extract backend data.",
        "Cross-Site Scripting (XSS)": "Injected client-side script attempted execution in victim browser context.",
        "DNS Tunneling": "DNS queries carried encoded data chunks to mimic legitimate resolver activity.",
        "SSH Bruteforce": "Automated credential guessing targeted SSH with rapid user/password permutations.",
        "Malware Beaconing": "Compromised endpoint performed periodic callback to suspicious infrastructure.",
        "C2 Keepalive Channel": "Persistent command channel maintained encrypted keepalive and task polling traffic.",
        "ICMP Abuse Sweep": "ICMP probes enumerated responsive hosts and routing behavior across the subnet.",
        "HTTP Exploit Attempt": "Web endpoint received exploit-like payloads targeting known parsing weaknesses.",
        "FTP Abuse": "FTP service used for unauthorized directory traversal and staged data movement.",
        "SMTP Abuse / Phishing Relay": "Mail relay behavior suggested phishing campaign staging and outbound abuse.",
        "Network Service Recon": "Adversary profiled internal services to identify lateral movement opportunities.",
        "Data Exfiltration over HTTPS": "Sensitive records were compressed and exfiltrated via encrypted web channel.",
    }
    tech = {
        "Port Scanning": "Pattern aligns with active scanning and network service discovery behavior documented in ATT&CK-style recon tactics.",
        "SYN Flood": "Traffic showed repeated SYN without handshake completion, matching denial strategy against TCP state tables.",
        "DDoS Traffic Burst": "Observed protocol abuse and packet amplification behavior resembles resource exhaustion campaigns.",
        "SQL Injection": "HTTP parameters carried SQL meta-characters, union clauses, and time-based probes indicating injection testing.",
        "Cross-Site Scripting (XSS)": "Payloads included script tags and event handlers intended for reflected or stored script execution.",
        "DNS Tunneling": "Long, randomized subdomain labels and TXT patterns indicate covert channel over DNS application layer.",
        "SSH Bruteforce": "Authentication telemetry indicated high-rate failures with credential stuffing and username enumeration.",
        "Malware Beaconing": "Regular interval callbacks with low payload variance suggest automated malware check-in logic.",
        "C2 Keepalive Channel": "Encrypted session reuse with deterministic timing and small tasking frames indicates C2 maintenance traffic.",
        "ICMP Abuse Sweep": "Echo request fan-out and timing regularity are consistent with host discovery and path probing.",
        "HTTP Exploit Attempt": "Request bodies and headers carried exploit markers and parser edge-case payloads targeting web middleware.",
        "FTP Abuse": "Control channel commands and transfer sequence indicate unauthorized file staging via legacy protocol exposure.",
        "SMTP Abuse / Phishing Relay": "SMTP transaction volume, sender spoofing, and campaign-like metadata indicate malicious relay usage.",
        "Network Service Recon": "Connection attempts against admin ports mapped service availability for potential lateral operations.",
        "Data Exfiltration over HTTPS": "Outbound TLS sessions transferred atypically large compressed objects to untrusted external destination.",
    }
    return short[a], tech[a]


def snort_rule(profile, sid, dst_port, indicator):
    proto = profile["protocol"]
    msg = profile["attack_name"]
    classtype = profile["classtype"]
    if proto == "icmp":
        return f'alert icmp any any -> $HOME_NET any (msg:"SOC {msg} activity"; itype:8; detection_filter:track by_src,count 25,seconds 5; classtype:{classtype}; sid:{sid}; rev:1;)'
    if proto == "dns":
        return f'alert udp any any -> $HOME_NET 53 (msg:"SOC {msg} suspected"; content:"|00 10 00 01|"; offset:2; depth:4; dsize:>180; detection_filter:track by_src,count 12,seconds 60; classtype:{classtype}; sid:{sid}; rev:1;)'
    if "SQL Injection" in msg:
        return f'alert tcp any any -> $HOME_NET {dst_port} (msg:"SOC SQLi pattern"; flow:to_server,established; content:"{random.choice(["UNION SELECT"," OR 1=1","information_schema","SLEEP("])}"; nocase; http_uri; classtype:{classtype}; sid:{sid}; rev:1;)'
    if "XSS" in msg:
        return f'alert tcp any any -> $HOME_NET {dst_port} (msg:"SOC XSS payload"; flow:to_server,established; content:"<script"; nocase; http_client_body; pcre:"/onerror\\s*=|onload\\s*=|javascript:/i"; classtype:{classtype}; sid:{sid}; rev:1;)'
    if "SSH Bruteforce" in msg:
        return f'alert tcp any any -> $HOME_NET 22 (msg:"SOC SSH brute force"; flow:to_server,established; content:"SSH-"; depth:4; detection_filter:track by_src,count 8,seconds 60; classtype:{classtype}; sid:{sid}; rev:1;)'
    if "SYN Flood" in msg:
        return f'alert tcp any any -> $HOME_NET {dst_port} (msg:"SOC SYN flood pattern"; flags:S; detection_filter:track by_dst,count 500,seconds 2; classtype:{classtype}; sid:{sid}; rev:1;)'
    if "DDoS" in msg:
        return f'alert udp any any -> $HOME_NET {dst_port} (msg:"SOC DDoS UDP burst"; dsize:>900; detection_filter:track by_dst,count 400,seconds 2; classtype:{classtype}; sid:{sid}; rev:1;)'
    if "Beaconing" in msg or "C2" in msg:
        return f'alert tcp $HOME_NET any -> any {dst_port} (msg:"SOC {msg} outbound"; flow:to_server,established; content:"POST"; http_method; detection_filter:track by_src,count 6,seconds 300; classtype:{classtype}; sid:{sid}; rev:1;)'
    if "SMTP" in msg:
        return f'alert tcp $HOME_NET any -> any {dst_port} (msg:"SOC SMTP abuse pattern"; flow:to_server,established; content:"MAIL FROM:"; nocase; detection_filter:track by_src,count 20,seconds 120; classtype:{classtype}; sid:{sid}; rev:1;)'
    if "FTP" in msg:
        return f'alert tcp any any -> $HOME_NET 21 (msg:"SOC FTP abuse command sequence"; flow:to_server,established; content:"USER "; nocase; content:"PASS "; distance:0; within:40; classtype:{classtype}; sid:{sid}; rev:1;)'
    if "HTTP Exploit" in msg or "Exfiltration" in msg:
        return f'alert tcp $HOME_NET any -> any {dst_port} (msg:"SOC {msg} suspicious outbound"; flow:to_server,established; content:"/api/export"; nocase; http_uri; dsize:>600; classtype:{classtype}; sid:{sid}; rev:1;)'
    return f'alert tcp any any -> $HOME_NET {dst_port} (msg:"SOC {msg} detection"; flags:S; detection_filter:track by_src,count 15,seconds 5; classtype:{classtype}; sid:{sid}; rev:1;)'


def rule_explanation(profile):
    base = {
        "Reconnaissance": "Rule emphasizes repeated probing cadence and protocol markers to surface host/service discovery without matching one packet only.",
        "DoS": "Rule uses short time-window counters to detect volumetric behavior and backlog exhaustion attempts.",
        "DDoS": "Rule captures high-rate bursts and payload-size anomalies to identify distributed availability attacks early.",
        "Web Attack": "Rule inspects HTTP URI/body with case-insensitive content and regex indicators for exploit primitives.",
        "C2/Exfiltration": "Rule tracks suspicious DNS application-layer characteristics tied to covert channels.",
        "Credential Access": "Rule counts repeated auth-like SSH handshakes from same source to flag brute-force automation.",
        "Malware Traffic": "Rule detects recurring outbound beacons and request consistency typical of malware callbacks.",
        "Command and Control": "Rule highlights periodic encrypted session behavior and low-volume command polling.",
        "Initial Access": "Rule targets exploit syntax in incoming web requests associated with edge service compromise attempts.",
        "Lateral Movement": "Rule watches legacy protocol command patterns often abused for unauthorized movement.",
        "Collection/Delivery": "Rule detects unusual SMTP command rates and relay behavior linked to phishing operations.",
        "Exfiltration": "Rule focuses on outbound high-volume encrypted transfers and suspicious export paths.",
    }
    return base.get(profile["attack_family"], "Rule correlates protocol semantics with attack timing and content indicators.")


def false_positive(profile):
    fam = profile["attack_family"]
    if fam in {"Reconnaissance", "Web Attack"}:
        return "Moderate: vulnerability scanners, QA tests, or aggressive monitoring tools may trigger similar patterns."
    if fam in {"DoS", "DDoS"}:
        return "Low to moderate: flash crowds or load testing can resemble attack bursts if baselines are not tuned."
    if fam in {"Credential Access"}:
        return "Moderate: failed logins from jump hosts or misconfigured automation can look similar."
    if fam in {"Collection/Delivery"}:
        return "Moderate: bulk mail systems and newsletter relays may overlap without sender reputation context."
    if fam in {"Exfiltration", "C2/Exfiltration", "Command and Control", "Malware Traffic"}:
        return "Low to moderate: backup agents, software updaters, or telemetry platforms can mimic periodic outbound traffic."
    return "Moderate: environment-specific legitimate behavior can overlap without asset role context."


def variants(profile):
    a = profile["attack_name"]
    return [
        f"{a} variant using alternate source subnet and slower timing profile",
        f"{a} variant with protocol camouflage and randomized packet size",
        f"{a} variant targeting backup infrastructure during off-hours",
    ]


def build_sample(i, sid_base=2100000):
    profile = ATTACK_PROFILES[i % len(ATTACK_PROFILES)]
    src_ip = rand_ip(private=False if profile["attack_family"] in {"Reconnaissance", "DDoS"} else random.choice([True, False]))
    dst_ip = rand_ip(private=True)
    src_port = random.randint(1024, 65535)
    dst_port = random.choice(profile["ports"])
    short_desc, tech_desc = attack_descriptions(profile)
    indicators = indicators_for(profile)
    context = (
        f"{random.choice(NETWORK_ZONES)} | target_asset={random.choice(ASSETS)} | "
        f"src_geo={random.choice(['US','DE','NL','BR','MA','SG'])} | "
        f"time_window={random.choice(['business-hours','off-hours','weekend'])}"
    )
    if "SMTP" in profile["attack_name"]:
        context += f" | lure_subject=\"{random.choice(MAIL_SUBJECTS)}\""

    nl = build_network_log(profile, src_ip, dst_ip, src_port, dst_port, i)
    sid = sid_base + i
    rule = snort_rule(profile, sid, dst_port, indicators[0])

    return {
        "id": f"CSDS-{i:04d}",
        "attack_name": profile["attack_name"],
        "attack_family": profile["attack_family"],
        "severity": profile["severity"],
        "protocol": profile["protocol"],
        "network_context": context,
        "attack_description": short_desc,
        "technical_description": tech_desc,
        "malicious_indicators": indicators,
        "network_log": nl,
        "snort_rule": rule,
        "rule_explanation": rule_explanation(profile),
        "false_positive_risk": false_positive(profile),
        "generated_variants": variants(profile),
    }


def validate(samples):
    required = {
        "id", "attack_name", "attack_family", "severity", "protocol", "network_context",
        "attack_description", "technical_description", "malicious_indicators", "network_log",
        "snort_rule", "rule_explanation", "false_positive_risk", "generated_variants",
    }
    for s in samples:
        if set(s.keys()) != required:
            raise ValueError("Schema mismatch")
        if not s["snort_rule"].startswith("alert "):
            raise ValueError(f"Bad rule prefix for {s['id']}")
        if "sid:" not in s["snort_rule"]:
            raise ValueError(f"Missing sid in {s['id']}")
        if not isinstance(s["malicious_indicators"], list) or len(s["malicious_indicators"]) < 2:
            raise ValueError(f"Indicators too weak in {s['id']}")


def sample_signature(s):
    # Noise control: consider a sample duplicate if these core detection signals match.
    return (
        s["attack_name"],
        s["protocol"],
        s["network_log"],
        s["snort_rule"],
    )


def generate_unique_samples(start_idx, count, existing=None):
    existing = existing or []
    sigs = {sample_signature(x) for x in existing}
    out = []
    i = start_idx
    while len(out) < count:
        s = build_sample(i)
        sig = sample_signature(s)
        if sig not in sigs:
            out.append(s)
            sigs.add(sig)
        i += 1
    return out


def write_outputs(samples):
    validate(samples)
    samples = sorted(samples, key=lambda x: x["id"])
    OUT_JSON.write_text(json.dumps(samples, indent=2))

    fieldnames = list(samples[0].keys())
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in samples:
            row2 = row.copy()
            row2["malicious_indicators"] = " | ".join(row2["malicious_indicators"])
            row2["generated_variants"] = " | ".join(row2["generated_variants"])
            writer.writerow(row2)


def main():
    samples = [build_sample(i + 1) for i in range(500)]
    write_outputs(samples)
    print(f"Wrote {len(samples)} samples")
    print(f"JSON: {OUT_JSON}")
    print(f"CSV : {OUT_CSV}")


def extend_dataset(extra_count=500):
    if OUT_JSON.exists():
        existing = json.loads(OUT_JSON.read_text())
    else:
        existing = []
    if not existing:
        existing = [build_sample(i + 1) for i in range(500)]

    max_id = 0
    max_sid = 2100000
    for s in existing:
        try:
            max_id = max(max_id, int(str(s["id"]).split("-")[1]))
            max_sid = max(max_sid, int(s["snort_rule"].split("sid:")[1].split(";")[0]))
        except Exception:
            continue

    new_samples = generate_unique_samples(max_id + 1, extra_count, existing=existing)

    # Re-sequence SIDs for appended data to avoid collisions.
    sid = max_sid + 1
    for s in new_samples:
        rule = s["snort_rule"]
        before = rule.split("sid:")[0]
        after = rule.split(";", 1)[1]
        s["snort_rule"] = f"{before}sid:{sid};{after}"
        sid += 1

    merged = existing + new_samples
    write_outputs(merged)
    print(f"Extended dataset by {len(new_samples)} samples")
    print(f"Total samples: {len(merged)}")
    print(f"JSON: {OUT_JSON}")
    print(f"CSV : {OUT_CSV}")


if __name__ == "__main__":
    extend_dataset(extra_count=500)
