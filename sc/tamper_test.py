"""Negative test: the edge server replaces one head's sketch after submission -> verifyBatch must revert."""
import json, numpy as np
from web3 import Web3, EthereumTesterProvider
from eth_tester import EthereumTester, PyEVMBackend
art = json.load(open('EECFGovernance.json')); K = 5
backend = PyEVMBackend(PyEVMBackend.generate_genesis_params(overrides={'gas_limit': 30_000_000}), PyEVMBackend.generate_genesis_state(num_accounts=K + 1))
w3 = Web3(EthereumTesterProvider(EthereumTester(backend)))
acc = [w3.eth.account.from_key(k.to_bytes()) for k in backend.account_keys]; edge, heads = acc[0], acc[1:K + 1]
def send(a, fn):
    tx = fn.build_transaction({'from': a.address, 'nonce': w3.eth.get_transaction_count(a.address), 'gas': 12_000_000, 'maxFeePerGas': 10**10, 'maxPriorityFeePerGas': 1})
    rc = w3.eth.wait_for_transaction_receipt(w3.eth.send_raw_transaction(a.sign_transaction(tx).raw_transaction)); return rc.status
C = w3.eth.contract(abi=art['abi'], bytecode=art['bytecode'])
tx = C.constructor().build_transaction({'from': edge.address, 'nonce': 0, 'gas': 5_000_000, 'maxFeePerGas': 10**10, 'maxPriorityFeePerGas': 1})
c = w3.eth.contract(address=w3.eth.wait_for_transaction_receipt(w3.eth.send_raw_transaction(edge.sign_transaction(tx).raw_transaction)).contractAddress, abi=art['abi'])
rng = np.random.default_rng(0); addrs = [h.address for h in heads]
pack = lambda v: [int.from_bytes(b''.join(int(x & 0xFFFF).to_bytes(2, 'big') for x in v[w*16:(w+1)*16]), 'big').to_bytes(32, 'big') for w in range(4)]
sks = [pack(rng.integers(-500, 500, 64)) for _ in range(K)]
send(edge, c.functions.elect(1, addrs, [1] * K))
for i, h in enumerate(heads): send(h, c.functions.submit(1, b'\x01' * 32, Web3.keccak(b''.join(sks[i]))))
bad = list(sks); bad[2] = pack(rng.integers(-500, 500, 64))
try:
    st = send(edge, c.functions.verifyBatch(1, addrs, bad)); res = 'tampered batch status %d' % st
except Exception as e: res = 'tampered batch reverted: ' + str(e)[:60]
print(res); print('honest batch status', send(edge, c.functions.verifyBatch(1, addrs, sks)))
# a batch that repeats one head and omits another must also revert
send(edge, c.functions.elect(2, addrs, [1] * K))
for i, h in enumerate(heads): send(h, c.functions.submit(2, b'\x02' * 32, Web3.keccak(b''.join(sks[i]))))
dup = [addrs[0]] + addrs[:-1]; dsk = [sks[0]] + sks[:-1]
try: st = send(edge, c.functions.verifyBatch(2, dup, dsk)); print('duplicate-head batch status', st)
except Exception as e: print('duplicate-head batch reverted')
print('honest batch status', send(edge, c.functions.verifyBatch(2, addrs, sks)))
