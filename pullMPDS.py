#def pullMPDS():
#    os.environ['MPDS_KEY'] = 'your_key'
#    client = MPDSDataRetrieval(dtype=MPDSDataTypes.PEER_REVIEWED)

#dilist = []
entries = []

for a in atoms:
    for c in classes:
        try:
            adata = client.get_data(
            {"classes": f"{c}", "elements": f"{a}", "props": "atomic structure"},
            fields={'S':['entry', 'chemical_formula', 'cell_abc', 'sg_n', 'basis_noneq', 'els_noneq']}
            )
        except:
            continue
        for item in adata:
            try:
                item[0]
            except:
                continue
            if not isinstance(item, list): continue
            if item[0] not in entries:
                print(item)
                entries.append(item[0])
                try:
                    crystal = MPDSDataRetrieval.compile_crystal(item, 'pymatgen')
                except:
                    continue
                if not crystal: continue


#            adata = client.get_data({el})
