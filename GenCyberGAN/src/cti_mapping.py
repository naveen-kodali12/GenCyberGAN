import json
import numpy as np

DEFAULT_CTI = {
    'Benign': ('Benign','Normal_Traffic','general'),
    'DoS_DDoS': ('Impact','Network_Denial_of_Service','tcp_udp_http'),
    'Reconnaissance': ('Discovery','Network_Service_Scanning','tcp_icmp'),
    'BruteForce': ('Credential_Access','Brute_Force','ssh_ftp_web'),
    'Botnet': ('Command_and_Control','Application_Layer_Protocol','tcp_http'),
    'WebAttack': ('Initial_Access','Exploit_Public_Facing_Application','http'),
    'Infiltration': ('Defense_Evasion','Exfiltration_Over_Web_Service','http_ssl'),
    'Exploit': ('Initial_Access','Exploit_Public_Facing_Application','multi_service'),
    'RareExploit': ('Execution','Exploitation_for_Client_Execution','shell_service'),
    'Theft': ('Collection','Data_from_Local_System','tcp')
}


def label_to_cti(label):
    return DEFAULT_CTI.get(str(label), ('Unknown','Unknown_Technique','unknown'))


def build_cti_vocab(labels):
    triples = [label_to_cti(l) for l in labels]
    tactics = sorted(set(t for t,_,_ in triples))
    techniques = sorted(set(te for _,te,_ in triples))
    services = sorted(set(s for *_,s in triples))
    return {'tactic': {v:i for i,v in enumerate(tactics)},
            'technique': {v:i for i,v in enumerate(techniques)},
            'service': {v:i for i,v in enumerate(services)}}


def encode_labels_to_cti_ids(labels, vocab):
    ids = []
    for lab in labels:
        t, te, s = label_to_cti(lab)
        ids.append([vocab['tactic'].get(t, 0), vocab['technique'].get(te, 0), vocab['service'].get(s, 0)])
    return np.asarray(ids, dtype='int64')


def save_cti_vocab(vocab, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=2)


def load_cti_vocab(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
