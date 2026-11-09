"""Measure per-function gas and serialised transaction size of EECFGovernance on py-evm.
Rounds 1..R with K heads; one Byzantine head (sign-flipped sketch) per round after round 1."""
import json, sys, numpy as np
from web3 import Web3, EthereumTesterProvider
from eth_tester import EthereumTester, PyEVMBackend

K_LIST = [int(x) for x in sys.argv[1:]] or [10]
art = json.load(open('EECFGovernance.json'))

def run(K, R=6, seed=0, timer=None):
    rng = np.random.default_rng(seed)
    backend = PyEVMBackend(PyEVMBackend.generate_genesis_params(overrides={'gas_limit': 30_000_000}),
                           PyEVMBackend.generate_genesis_state(num_accounts=K + 1))
    t = EthereumTester(backend)
    w3 = Web3(EthereumTesterProvider(t))
    keys = backend.account_keys
    accts = [w3.eth.account.from_key(k.to_bytes()) for k in keys]
    edge = accts[0]; heads = accts[1:K + 1]
    C = w3.eth.contract(abi=art['abi'], bytecode=art['bytecode'])
    def send(acct, fn):
        tx = fn.build_transaction({'from': acct.address, 'nonce': w3.eth.get_transaction_count(acct.address),
                                   'gas': 12_000_000, 'maxFeePerGas': 10**10, 'maxPriorityFeePerGas': 1})
        st = acct.sign_transaction(tx)
        raw = st.raw_transaction if hasattr(st, 'raw_transaction') else st.rawTransaction
        h = w3.eth.send_raw_transaction(raw)
        rc = w3.eth.wait_for_transaction_receipt(h)
        assert rc.status == 1, fn
        return rc.gasUsed, len(raw)
    # deploy
    tx = C.constructor().build_transaction({'from': edge.address, 'nonce': 0, 'gas': 5_000_000,
                                            'maxFeePerGas': 10**10, 'maxPriorityFeePerGas': 1})
    st = edge.sign_transaction(tx); raw = st.raw_transaction if hasattr(st, 'raw_transaction') else st.rawTransaction
    rc = w3.eth.wait_for_transaction_receipt(w3.eth.send_raw_transaction(raw))
    deploy_gas = rc.gasUsed
    c = w3.eth.contract(address=rc.contractAddress, abi=art['abi'])
    g = rng.integers(-1000, 1000, 64)
    rows = []
    import time
    for r in range(1, R + 1):
        t0 = time.perf_counter()
        rec = {'round': r}
        addrs = [h.address for h in heads]
        rec['elect'] = send(edge, c.functions.elect(r, addrs, [int(x) for x in rng.integers(1, 10**6, K)]))
        # 64-dim int16 sketches packed into 4 x bytes32; head 0 is Byzantine (sign-flipped) after round 1
        def pack(v):
            v = np.asarray(v, dtype=np.int64) & 0xFFFF
            words = []
            for w_ in range(4):
                x = 0
                for j in range(16): x = (x << 16) | int(v[w_ * 16 + j])
                words.append(x.to_bytes(32, 'big'))
            return words
        sks = []
        for i in range(K):
            s_ = np.clip(g + rng.integers(-300, 300, 64), -32768, 32767)
            if r > 1 and i == 0: s_ = -s_
            sks.append(pack(s_))
        skh = [Web3.keccak(b''.join(sk_)) for sk_ in sks]
        sub = [send(h, c.functions.submit(r, bytes(rng.integers(0, 256, 32, dtype=np.uint8)), skh[i])) for i, h in enumerate(heads)]
        gas, size = send(edge, c.functions.verifyBatch(r, addrs, sks))
        ver = [(gas, size)]
        acc = sum(c.functions.acceptedRound(h.address).call() == r for h in heads)
        rec['distribute'] = send(edge, c.functions.distribute(r, addrs, [int(x) for x in rng.integers(1, 1000, K)],
                                  [int(x) for x in rng.integers(1, 1000, K)], 10**18,
                                  bytes(rng.integers(0, 256, 32, dtype=np.uint8))))
        rec['submit'] = (float(np.mean([x[0] for x in sub])), float(np.mean([x[1] for x in sub])))
        rec['verify'] = ver[0]
        rec['accepted'] = acc
        rec['total_gas'] = rec['elect'][0] + sum(x[0] for x in sub) + sum(x[0] for x in ver) + rec['distribute'][0]
        rec['total_bytes'] = rec['elect'][1] + sum(x[1] for x in sub) + sum(x[1] for x in ver) + rec['distribute'][1]
        rec['wall'] = time.perf_counter() - t0
        rows.append(rec)
    return deploy_gas, rows


def timed_round(K, rep):
    _, rows = run(K, R=rep + 1)
    w = [r['wall'] for r in rows[1:]]
    import numpy as np
    return float(np.mean(w)), float(np.std(w, ddof=1))
