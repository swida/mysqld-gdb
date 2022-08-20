import gdb

class AutoCVar(object):
    maxlen = 2
    gdb_can_set_cvar = True
    def __init__(self):
        self.cur_seq = ['a']
        self.gdb_can_set_cvar = hasattr(gdb, 'set_convenience_variable')

    def get_name(self):
        cname = ''.join(self.cur_seq)
        clen = len(self.cur_seq)
        for i, c in reversed(list(enumerate(self.cur_seq))):
            if c == 'z':
                continue
            self.cur_seq[i] = chr(ord(c) + 1)
            for j in range(i + 1, clen):
                self.cur_seq[j] = 'a'
            break
        else:
            self.cur_seq = ['a'] * \
                (1 if clen == self.maxlen else (clen + 1))

        return cname

    def set_nvar(self, varname, var):
        if not self.gdb_can_set_cvar:
            return ''
        gdb.set_convenience_variable(varname, var)
        return '$' + varname

    def set_var(self, var):
        return self.set_nvar(self.get_name(), var)

autocvar = AutoCVar()

class AutoNumCVar(object):
    def __init__(self, init_num = 0):
        self.cvar_name = autocvar.get_name()
        self.cur_num = init_num

    def set_var(self, var):
        cvname = self.cvar_name + str(self.cur_num)
        self.cur_num += 1
        return autocvar.set_nvar(cvname, var)

def get_name():
    return autocvar.get_name()

def set_var(var):
    return autocvar.set_var(var)

def set_nvar(varname, var):
    return autocvar.set_nvar(varname, var)

__all__ = ['autocvar', 'AutoNumCVar', 'get_name', 'set_var', 'set_nvar']
