// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title EECF-IoT governance contract (Tier 3)
/// @notice Cluster-head election record, gradient-hash submission, cosine-similarity
///         Byzantine screening on a 64-dimensional quantised sketch of each update
///         (Eq. 15), and proportional incentive settlement (60 % energy efficiency,
///         40 % training-set diversity). Deployed on a permissioned EVM chain
///         (Hyperledger Besu, QBFT); only the registered edge aggregator may call
///         elect / verify / distribute.
contract EECFGovernance {
    uint256 public constant D = 64;            // sketch dimension
    uint256 public constant THETA_SQ = 400;    // theta_byz^2 * 1e4  (theta = 0.20)

    address public immutable edge;             // registered edge aggregator
    uint64  public currentRound;

    mapping(address => uint64)  public headRound;     // round in which an address is a head
    mapping(address => bytes32) public updateHash;    // hash (IPFS CID digest) of last update
    mapping(address => bytes32) public sketchHash;    // head's commitment to its own 64-dim sketch
    mapping(address => uint64)  public acceptedRound; // round in which update was accepted
    mapping(address => uint256) public balance;       // incentive tokens
    mapping(uint64  => bytes32) public globalModel;   // global model hash per round
    mapping(uint64  => uint256) public electedCount;  // number of heads elected for a round
    mapping(address => uint64)  public verifiedRound; // round in which a head was screened (once)
    mapping(address => uint64)  public paidRound;     // round in which a head was settled (once)

    event Elected(uint64 indexed round, address[] heads, uint32[] scores);
    event Submitted(uint64 indexed round, address indexed head, bytes32 hash, bytes32 sketchHash);
    event Verified(uint64 indexed round, address indexed head, bool accepted);
    event Distributed(uint64 indexed round, bytes32 globalModelHash, uint256 paid);

    modifier onlyEdge() { require(msg.sender == edge, "edge only"); _; }

    constructor() { edge = msg.sender; }

    /// (1) record the cluster-head set chosen by Algorithm 1 for this round
    function elect(uint64 round, address[] calldata heads, uint32[] calldata scores) external onlyEdge {
        require(heads.length == scores.length, "len");
        require(round > currentRound, "round");
        currentRound = round;
        for (uint256 i = 0; i < heads.length; ++i) {
            require(headRound[heads[i]] != round, "duplicate head");
            headRound[heads[i]] = round;
        }
        electedCount[round] = heads.length;
        emit Elected(round, heads, scores);
    }

    /// (2) a cluster head anchors the hash of its (off-chain, IPFS-stored) model update and
    ///     commits to the sketch of that update; verifyBatch() only accepts the committed sketch,
    ///     so the edge server cannot substitute or alter a head's sketch.
    function submit(uint64 round, bytes32 h, bytes32 skHash) external {
        require(headRound[msg.sender] == round && round == currentRound, "not head");
        updateHash[msg.sender] = h;
        sketchHash[msg.sender] = skHash;
        emit Submitted(round, msg.sender, h, skHash);
    }

    /// (3) cosine-similarity screening (Eq. 15) of all K updates of a round in one call.
    ///     Each update is represented by a D=64 random-projection sketch (int16, packed
    ///     4 x bytes32 per head). The reference g-bar is the coordinate-wise median of the
    ///     K sketches, computed on-chain; an update is accepted iff cos(s_k, g-bar) >= theta.
    function verifyBatch(uint64 round, address[] calldata heads, bytes32[4][] calldata sk)
        external onlyEdge returns (uint256 accepted)
    {
        uint256 K = heads.length;
        require(K == sk.length && K > 0, "len");
        require(K == electedCount[round], "every elected head exactly once");
        int256[] memory v = new int256[](K * D);                 // decoded sketches, row-major
        unchecked {
            for (uint256 k = 0; k < K; ++k)
                for (uint256 w = 0; w < 4; ++w) {
                    uint256 word = uint256(sk[k][w]);
                    for (uint256 j = 0; j < 16; ++j)
                        v[k * D + w * 16 + j] = int256(int16(uint16(word >> ((15 - j) * 16))));
                }
            int256[] memory med = new int256[](D);
            int256[] memory col = new int256[](K);
            for (uint256 d = 0; d < D; ++d) {
                for (uint256 k = 0; k < K; ++k) {                 // insertion sort of column d
                    int256 x = v[k * D + d]; uint256 j = k;
                    while (j > 0 && col[j - 1] > x) { col[j] = col[j - 1]; --j; }
                    col[j] = x;
                }
                med[d] = (K % 2 == 1) ? col[K / 2] : (col[K / 2 - 1] + col[K / 2]) / 2;
            }
            int256 n2;
            for (uint256 d = 0; d < D; ++d) n2 += med[d] * med[d];
            for (uint256 k = 0; k < K; ++k) {
                address h = heads[k];
                require(headRound[h] == round && updateHash[h] != bytes32(0), "no update");
                require(verifiedRound[h] != round, "duplicate head");
                verifiedRound[h] = round;
                require(keccak256(abi.encode(sk[k])) == sketchHash[h], "sketch mismatch");
                int256 dot; int256 n1;
                for (uint256 d = 0; d < D; ++d) { int256 a = v[k * D + d]; dot += a * med[d]; n1 += a * a; }
                bool ok = n2 == 0 || (dot > 0 && dot * dot * 10000 >= int256(THETA_SQ) * n1 * n2);
                if (ok) { acceptedRound[h] = round; ++accepted; }
                emit Verified(round, h, ok);
            }
        }
    }

    /// (4) settle incentives for accepted heads and anchor the new global model
    function distribute(uint64 round, address[] calldata heads, uint32[] calldata energyEff,
                        uint32[] calldata diversity, uint256 pool, bytes32 modelHash) external onlyEdge {
        uint256 se; uint256 sd; uint256 n = heads.length;
        require(n == electedCount[round] && n == energyEff.length && n == diversity.length, "every elected head exactly once");
        for (uint256 i = 0; i < n; ++i) {
            require(headRound[heads[i]] == round && paidRound[heads[i]] != round, "duplicate or non-head");
            paidRound[heads[i]] = round;
        }
        for (uint256 i = 0; i < n; ++i) if (acceptedRound[heads[i]] == round) { se += energyEff[i]; sd += diversity[i]; }
        uint256 paid;
        for (uint256 i = 0; i < n; ++i) {
            if (acceptedRound[heads[i]] != round) continue;
            uint256 amt = (se > 0 ? pool * 60 * energyEff[i] / (100 * se) : 0)
                        + (sd > 0 ? pool * 40 * diversity[i] / (100 * sd) : 0);
            balance[heads[i]] += amt; paid += amt;
        }
        globalModel[round] = modelHash;
        emit Distributed(round, modelHash, paid);
    }
}
