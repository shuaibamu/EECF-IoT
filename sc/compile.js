const solc=require('solc'),fs=require('fs');
const input={language:'Solidity',sources:{'EECFGovernance.sol':{content:fs.readFileSync('EECFGovernance.sol','utf8')}},
settings:{viaIR:true,optimizer:{enabled:true,runs:200},evmVersion:'paris',outputSelection:{'*':{'*':['abi','evm.bytecode.object']}}}};
const out=JSON.parse(solc.compile(JSON.stringify(input)));
if(out.errors) for(const e of out.errors) console.error(e.formattedMessage);
const c=out.contracts['EECFGovernance.sol'].EECFGovernance;
fs.writeFileSync('EECFGovernance.json',JSON.stringify({abi:c.abi,bytecode:c.evm.bytecode.object}));
console.log('version',solc.version(),'bytecode bytes',c.evm.bytecode.object.length/2);
