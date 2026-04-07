// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract ModelRegistry {

    // Stores one update per client per round
    struct ModelUpdate {
        address client;       // who submitted
        bytes32 updateHash;   // hash of the model weights
        uint256 roundNumber;  // which FL round
        bool isVerified;      // did it pass the check?
        uint256 timestamp;
    }

    // All updates stored here, accessible to everyone
    ModelUpdate[] public updates;

    // Emitted whenever a valid update is logged
    event UpdateLogged(address indexed client, bytes32 updateHash, uint256 round);

    // Called by each client after local training
    function submitUpdate(bytes32 _updateHash, uint256 _roundNumber) public {
        // Basic check: hash must not be empty (zero)
        require(_updateHash != bytes32(0), "Invalid update hash");

        updates.push(ModelUpdate({
            client:      msg.sender,
            updateHash:  _updateHash,
            roundNumber: _roundNumber,
            isVerified:  true,
            timestamp:   block.timestamp
        }));

        emit UpdateLogged(msg.sender, _updateHash, _roundNumber);
    }

    // Server calls this to fetch all verified updates for a given round
    function getVerifiedUpdates(uint256 _roundNumber)
        public view returns (address[] memory, bytes32[] memory)
    {
        uint256 count = 0;
        for (uint i = 0; i < updates.length; i++) {
            if (updates[i].roundNumber == _roundNumber && updates[i].isVerified)
                count++;
        }

        address[] memory clients = new address[](count);
        bytes32[] memory hashes  = new bytes32[](count);
        uint256 idx = 0;

        for (uint i = 0; i < updates.length; i++) {
            if (updates[i].roundNumber == _roundNumber && updates[i].isVerified) {
                clients[idx] = updates[i].client;
                hashes[idx]  = updates[i].updateHash;
                idx++;
            }
        }
        return (clients, hashes);
    }

    // How many updates have been submitted total?
    function getTotalUpdates() public view returns (uint256) {
        return updates.length;
    }
}