import re
import pandas as pd

CANONICAL_ALIASES = {
    'timestamp': ['timestamp','time','stime','ltime','flow start time','starttime','ts'],
    'src_ip': ['srcip','src_ip','source ip','source_ip','source','id.orig_h'],
    'dst_ip': ['dstip','dst_ip','destination ip','destination_ip','destination','id.resp_h'],
    'src_port': ['sport','src_port','source port','source_port','id.orig_p'],
    'dst_port': ['dsport','dport','dst_port','destination port','destination_port','id.resp_p'],
    'proto': ['proto','protocol'],
    'service': ['service','app_proto','application_protocol'],
    'state': ['state','conn_state','connection state'],
    'duration': ['dur','duration','flow duration'],
    'bytes': ['bytes','totbytes','total bytes','flow bytes/s','sbytes','dbytes'],
    'packets': ['packets','totpkts','total packets','spkts','dpkts'],
    'label': ['label','attack_cat','attack category','class','category','subcategory']
}


def clean_col(c: str) -> str:
    return re.sub(r'\s+', ' ', str(c).strip().lower().replace('-', ' ').replace('_', ' '))


def harmonize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    reverse = {}
    for canon, aliases in CANONICAL_ALIASES.items():
        for a in aliases:
            reverse[clean_col(a)] = canon
    rename = {}
    used = set()
    for col in df.columns:
        key = clean_col(col)
        if key in reverse and reverse[key] not in used:
            rename[col] = reverse[key]
            used.add(reverse[key])
        else:
            rename[col] = str(col).strip().replace(' ', '_')
    return df.rename(columns=rename)


def harmonize_attack_label(value):
    s = str(value).strip().lower()
    if s in {'0','normal','benign','background','nan','none'}:
        return 'Benign'
    if 'ddos' in s or 'dos' in s or 'denial' in s:
        return 'DoS_DDoS'
    if 'recon' in s or 'scan' in s or 'portsweep' in s or 'portscan' in s or 'analysis' in s:
        return 'Reconnaissance'
    if 'brute' in s or 'ftp' in s or 'ssh' in s or 'password' in s:
        return 'BruteForce'
    if 'bot' in s or 'c&c' in s or 'command' in s:
        return 'Botnet'
    if 'web' in s or 'sql' in s or 'xss' in s:
        return 'WebAttack'
    if 'infil' in s:
        return 'Infiltration'
    if 'shellcode' in s or 'worm' in s:
        return 'RareExploit'
    if 'exploit' in s or 'backdoor' in s or 'generic' in s or 'fuzzer' in s:
        return 'Exploit'
    if 'theft' in s:
        return 'Theft'
    return s.title().replace(' ', '_')
