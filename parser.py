def load_input_file(filename):

    variables = {}
    
    with open(filename, 'r') as f:
        for line in f:
    
            line = line.strip()
            if not line or line.startswith("#"):
                continue  # skip empty lines and comments
            if '=' not in line:
                raise ValueError(f"Line does not contain '=': {line}")
            key, value = map(str.strip, line.split('=', 1))
            
            # Handle special variables
            # TODO: now 'm' is also a "special variable"
            if key in ("frac_list", "m"):
                variables[key] = [float(v) for v in value.split()]
            elif key in ("sub_an_list", "ada_an_list"):
                variables[key] = value.split()
            elif value.replace('.', '', 1).replace('-', '', 1).isdigit():
                # int or float
                if '.' in value or 'e' in value.lower():
                    variables[key] = float(value)
                else:
                    variables[key] = int(value)
            else:
                # string
                variables[key] = value

    # Check on the length of input strings
    assert len(variables['frac_list'])==variables['na_ada'], "Adatom fractions not matching number of adatom types"
    assert len(variables['m'])==variables['na_ada'], "Adatom mass vector not matching number of adatom types"
    assert len(variables['ada_an_list'])==variables['na_ada'], "Adatom type list not matching number of adatom types"
    assert len(variables['sub_an_list'])==variables['na_sub'], "Substrate type list not matching number of substrate types"

    return variables