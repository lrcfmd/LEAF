from pymatgen.ext.matproj import MPRester
API_KEY = "8E8EtFRXXZiRAKD4Luvm7bvsUlcdm5ax"
# Get the structures of the 4 chosen structure types.

struct_files = ["cscl.cif", "rock_salt.cif", "zinc_blende.cif", "wurtzite.cif"]

with MPRester(API_KEY) as mpr:
        cscl_struct = mpr.get_structure_by_material_id("mp-22865")
        rock_salt_struct = mpr.get_structure_by_material_id("mp-22862")
        zinc_blende_struct = mpr.get_structure_by_material_id("mp-10695")
        wurtzite_struct = mpr.get_structure_by_material_id("mp-560588")
        # Save structures to cifs
        cscl_struct.to(filename="cscl.cif")
        rock_salt_struct.to(filename="rock_salt.cif")
        zinc_blende_struct.to(filename="zinc_blende.cif")
        wurtzite_struct.to(filename="wurtzite.cif")
