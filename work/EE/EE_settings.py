# Data Definitions
role_types = [
    'obj-per',  # 0
    'amount',   # 1
    'title',    # 2
    'sub-org',  # 3
    'number',   # 4
    'way',      # 5
    'collateral',   # 6
    'obj',      # 7
    'target-company',   # 8
    'share-org',    # 9
    'sub-per',  # 10
    'sub',      # 11
    'data',     # 12
    'obj-org',  # 13
    'proportion',   # 14
    'date',     # 15
    'share-per',    # 16
    'institution',  # 17
    'money'     # 18
]
role_index = {v: i for i, v in enumerate(role_types)}
event_types_initial = [
    '质押',
    '股份股权转让',
    '起诉',
    '投资',
    '减持'
]
event_types_initial_index = {v: i for i, v in enumerate(event_types_initial)}
event_types_full = [
    '质押',
    '股份股权转让',
    '起诉',
    '投资',
    '减持',
    '收购',
    '担保',
    '中标',
    '签署合同',
    '判决'
]
event_types_full_index = {v: i for i, v in enumerate(event_types_full)}


event_available_roles = {
    '质押': {'sub-org', 'sub-per', 'obj-org', 'obj-per', 'collateral', 'date', 'money', 'number', 'proportion'},
    '股份股权转让': {'sub-org', 'sub-per', 'obj-org', 'obj-per', 'collateral', 'date', 'money', 'number', 'proportion',
               'target-company'},
    '起诉': {'sub-org', 'sub-per', 'obj-org', 'obj-per', 'date'},
    '投资': {'sub', 'obj', 'money', 'date'},
    '减持': {'sub', 'obj', 'title', 'date', 'share-per', 'share-org'},
    '收购': {'sub-org', 'sub-per', 'obj-org', 'way', 'date', 'money', 'number', 'proportion', 'target-company'},
    '担保': {'sub-org', 'sub-per', 'obj-org', 'way', 'amount', 'date'},
    '中标': {'sub', 'obj', 'amount', 'date'},
    '签署合同': {'sub-org', 'sub-per', 'obj-org', 'obj-per', 'amount', 'date'},
    '判决': {'institution', 'sub-org', 'sub-per', 'obj-org', 'obj-per', 'date', 'money'}
}

