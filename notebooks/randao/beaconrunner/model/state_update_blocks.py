import model.parts.basic_duties as duties

state_update_blocks = [
    {
        'label': 'Attestations',
        'policies': {
            'action': duties.attest_policy
        },
        'variables': {
            'network': duties.update_attestations
        }
    },
    {
        'label': 'Sync committee',
        'policies': {
            'action': duties.sync_committee_policy
        },
        'variables': {
            'network': duties.update_sync_committees
        }
    },
    {
        'label': 'Block proposals',
        'policies': {
            'action': duties.propose_policy
        },
        'variables': {
            'network': duties.update_blocks
        }
    },
    {
        'label': 'Simulation tick',
        'policies': {
        },
        'variables': {
            'network': duties.tick
        }
    },
]
