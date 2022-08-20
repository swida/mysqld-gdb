"""A gdb module for mysql-server development which includes some mysql
commands and some pretty printers.

Commands:
mysql digest -- decode a statement digest to readable string
mysql item -- explore expression (Item) tree
mysql queryblock -- explore query block tree
mysql seltree -- explore SEL_TREE struct
mysql tablelist -- traverse TABLE_LIST list

Pretty printers of structs:
List
mem_root_deque
"""
from __future__ import print_function # python2.X support
import re
from autocvar import autocvar, AutoNumCVar
#
# Some utility functions
#
def std_vector_to_list(std_vector):
    """convert std::vector to a list"""
    out_list = []
    value_reference = std_vector['_M_impl']['_M_start']
    while value_reference != std_vector['_M_impl']['_M_finish']:
        out_list.append(value_reference.dereference())
        value_reference += 1

    return out_list

def mem_root_deque_to_list(deque):
    """convert mem_root_deque to a list"""
    out_list = []
    elttype = deque.type.template_argument(0)
    elements = deque['block_elements']
    start = deque['m_begin_idx']
    end = deque['m_end_idx']
    blocks = deque['m_blocks']

    p = start
    while p != end:
        elt = blocks[p / elements]['elements'][p % elements]
        out_list.append(elt)
        p += 1

    return out_list

#
# Some convenience variables for debug easily  because they are macros
#
autocvar.set_nvar('MAX_TABLES', gdb.parse_and_eval('sizeof(unsigned long long) * 8 - 3'))
autocvar.set_nvar('INNER_TABLE_BIT', gdb.parse_and_eval('((unsigned long long)1) << ($MAX_TABLES + 0)'))
autocvar.set_nvar('OUTER_REF_TABLE_BIT', gdb.parse_and_eval('((unsigned long long)1) << ($MAX_TABLES + 1)'))
autocvar.set_nvar('RAND_TABLE_BIT', gdb.parse_and_eval('((unsigned long long)1) << ($MAX_TABLES + 2)'))
autocvar.set_nvar('PSEUDO_TABLE_BITS', gdb.parse_and_eval('($INNER_TABLE_BIT | $OUTER_REF_TABLE_BIT | $RAND_TABLE_BIT)'))

# Define a mysql command prefix for all mysql related command
gdb.Command('mysql', gdb.COMMAND_DATA, prefix=True)

#
# Commands start here
#

#
# Some small utils
#
class DigestPrinter(gdb.Command):
    """decode a statement digest to readable string"""
    def __init__ (self):
        super (DigestPrinter, self).__init__ ("mysql digest", gdb.COMMAND_OBSCURE)

    def invoke(self, arg, from_tty):
        start_addr = gdb.parse_and_eval(arg)
        i = 0
        while i < 32:
            char_code = int(start_addr[i].cast(gdb.lookup_type("unsigned char")))
            print("%02x" % char_code, end='')
            i += 1
        print()
DigestPrinter()

#
# threads overview/search for mysql
#
def gdb_threads():
    if hasattr(gdb, 'selected_inferior'):
        threads = gdb.selected_inferior().threads()
    else:
        threads = gdb.inferiors()[0].threads()
    return threads

def pretty_frame_name(frame_name):
    """omit some stdc++ stacks"""
    pretty_names = (
        ('std::__invoke_impl', ''),
        ('std::__invoke', ''),
        ('std::_Bind', ''),
        ('Runnable::operator()', ''),
        ('std::thread::_Invoker', ''),
        ('std::thread::_State_impl', 'std::thread'),
        ('std::this_thread::sleep_for', 'std..sleep_for'))

    for templ, val in pretty_names:
        if frame_name.startswith(templ):
            return val

    return frame_name

def brief_backtrace(filter_threads):
    frames = ''
    frame = gdb.newest_frame() if hasattr(gdb, 'newest_frame') else gdb.selected_frame()
    while frame is not None:
        frame_name = frame.name() if frame.name() is not None else '??'
        if filter_threads is not None and frame_name in filter_threads:
            return None
        frame_name = pretty_frame_name(frame_name)
        if frame_name:
            frames += frame_name + ','
        frame = frame.older()
    frames = frames[:-1]
    return frames

class ThreadSearch(gdb.Command):
    """find threads given a regex which matchs thread name, parameter name or value"""

    def __init__ (self):
        super (ThreadSearch, self).__init__ ("thread search", gdb.COMMAND_OBSCURE)

    def invoke (self, arg, from_tty):
        pattern = re.compile(arg)
        threads = gdb_threads()
        old_thread = gdb.selected_thread()
        for thr in threads:
            thr.switch()
            backtrace = gdb.execute('bt', False, True)
            matched_frames = [fr for fr in backtrace.split('\n') if pattern.search(fr) is not None]
            if matched_frames:
                print(thr.num, brief_backtrace(None))

        old_thread.switch()
ThreadSearch()

class ThreadOverview(gdb.Command):
    """print threads overview, display all frames in one line and function name only for each frame"""
    # filter Innodb backgroud workers
    filter_threads = (
        # Innodb backgroud threads
        'log_closer',
        'buf_flush_page_coordinator_thread',
        'log_writer',
        'log_flusher',
        'log_write_notifier',
        'log_flush_notifier',
        'log_checkpointer',
        'lock_wait_timeout_thread',
        'srv_error_monitor_thread',
        'srv_monitor_thread',
        'buf_resize_thread',
        'buf_dump_thread',
        'dict_stats_thread',
        'fts_optimize_thread',
        'srv_purge_coordinator_thread',
        'srv_worker_thread',
        'srv_master_thread',
        'io_handler_thread',
        'event_scheduler_thread',
        'compress_gtid_table',
        'ngs::Scheduler_dynamic::worker_proxy'
        )
    def __init__ (self):
        super (ThreadOverview, self).__init__ ("thread overview", gdb.COMMAND_OBSCURE)

    def invoke (self, arg, from_tty):
        threads = gdb_threads()
        old_thread = gdb.selected_thread()
        thr_dict = {}
        for thr in threads:
            thr.switch()
            bframes = brief_backtrace(self.filter_threads)
            if bframes is None:
                continue
            if bframes in thr_dict:
                thr_dict[bframes].append(thr.num)
            else:
                thr_dict[bframes] = [thr.num,]
        thr_ow = [(v,k) for k,v in thr_dict.items()]
        thr_ow.sort(key = lambda l:len(l[0]), reverse=True)
        for nums_thr,funcs in thr_ow:
           print(','.join([str(i) for i in nums_thr]), funcs)
        old_thread.switch()
ThreadOverview()

class TreeWalker(object):
    """A base class for tree traverse"""

    def __init__(self):
        self.level_graph = []
        self.autoncvar = None
        self.current_level = 0

    def reset(self):
        self.level_graph = []
        self.autoncvar = AutoNumCVar()

    def walk(self, expr):
        self.reset()
        self.do_walk(expr, 0)

    def do_walk(self, expr, level):
        expr_typed = expr.dynamic_type
        expr_casted = expr.cast(expr_typed)
        expr_nodetype = None
        try:
            expr_nodetype = expr.type.template_argument(0)
            if expr_nodetype.code != gdb.TYPE_CODE_PTR:
                expr_nodetype = expr.type.template_argument(0).pointer()
        except (gdb.error, RuntimeError):
            expr_nodetype = None
            pass
        self.current_level = level
        level_graph = '  '.join(self.level_graph[:level])
        for i, c in enumerate(self.level_graph):
            if c == '`':
                self.level_graph[i] = ' '
        cname = self.autoncvar.set_var(expr_casted)
        left_margin = "{}{}".format('' if level == 0 else '--', cname)
        element_show_info = ''
        show_func = self.get_show_func(expr_typed, expr_nodetype)
        if show_func is not None:
            element_show_info = show_func(expr_casted)
        if element_show_info is not None:
            print("{}{} ({}) {} {}".format(
                  level_graph, left_margin, expr_typed, expr, element_show_info))
        walk_func = self.get_walk_func(expr_typed, expr_nodetype)
        if walk_func is None:
            return
        children = walk_func(expr_casted)
        if not children:
            return
        if len(self.level_graph) < level + 1:
            self.level_graph.append('|')
        else:
            self.level_graph[level] = '|'
        for i, child in enumerate(children):
            if i == len(children) - 1:
                self.level_graph[level] = '`'
            self.do_walk(child, level + 1)

    def get_action_func(self, element_type, action_prefix):
        def type_name(typ):
            return typ.name if typ.name != None and hasattr(typ, 'name') else str(typ)
        func_name = action_prefix + type_name(element_type)
        if hasattr(self, func_name):
            return getattr(self, func_name)

        for field in element_type.fields():
            if not field.is_base_class:
                continue
            typ = field.type
            func_name = action_prefix + type_name(typ)

            if hasattr(self, func_name):
                return getattr(self, func_name)

            return self.get_action_func(typ, action_prefix)
        return None

    def get_walk_func(self, element_type, element_type_templ):
        if element_type_templ != None:
            return self.get_action_func(element_type_templ.target(), 'walk_templ_')
        else:
            return self.get_action_func(element_type.target(), 'walk_')

    def get_show_func(self, element_type, element_type_templ):
        if element_type_templ != None:
            return self.get_action_func(element_type_templ.target(), 'show_templ_')
        else:
            return self.get_action_func(element_type.target(), 'show_')

class ItemDisplayer(object):
    """mysql item basic show functions"""
    def show_Item_ident(self, item):
        db_cata = []
        if item['db_name']:
            db_cata.append(item['db_name'].string())
        if item['table_name']:
            db_cata.append(item['table_name'].string())
        if item['field_name']:
            db_cata.append(item['field_name'].string())
        return 'field = ' + '.'.join(db_cata)

    def show_Item_int(self, item):
        return 'value = ' + str(item['value'])

    show_Item_float = show_Item_int

    def show_Item_string(self, item):
        return 'value = ' + item['str_value']['m_ptr'].string()

    def show_Item_decimal(self, item):
        sym = gdb.lookup_global_symbol('Item_decimal::val_real()')
        result = sym.value()(item)
        return 'value = ' + str(result)

    def show_uchar(self, item):
        return item

class ItemExpressionTraverser(gdb.Command, TreeWalker, ItemDisplayer):
    """explore expression (Item) tree"""

    def __init__ (self):
        super(self.__class__, self).__init__("mysql item", gdb.COMMAND_OBSCURE)

    def invoke(self, arg, from_tty):
        if not arg:
            print("usage: mysql item [Item]")
            return
        expr = gdb.parse_and_eval(arg)
        self.walk(expr)

    #
    # walk and show functions for each Item class
    #

    def walk_Item_func(self, val):
        children = []
        for i in range(val['arg_count']):
            children.append(val['args'][i])
        return children

    walk_Item_sum = walk_Item_func

    def walk_Item_cond(self, val):
        end_of_list = gdb.parse_and_eval('end_of_list').address
        item_list = val['list']
        nodetype = item_list.type.template_argument(0)
        cur_elt = item_list['first']
        children = []
        while cur_elt != end_of_list:
            info = cur_elt.dereference()['info']
            children.append(info.cast(nodetype.pointer()))
            cur_elt = cur_elt.dereference()['next']
        return children

ItemExpressionTraverser()

def print_TABLE_LIST(lt, autoncvar):
    table_name = lt['table_name'].string()
    alias = lt['alias'].string()
    tl_cnname = autoncvar.set_var(lt)
    s = '(' + tl_cnname + ')' + table_name
    return s if table_name == alias else s + " " + alias

def traverse_TABLE_LIST_low(table_list, next_field):
    tables = ''
    autoncvar = AutoNumCVar()
    while table_list:
        lt = table_list.dereference()
        tables += print_TABLE_LIST(lt, autoncvar) + ', '
        table_list = lt[next_field]

    if tables:
        tables = tables[0 : len(tables) - 2]

    return tables

def traverse_TABLE_LIST(table_list, only_leaf):
    s = traverse_TABLE_LIST_low(table_list, 'next_leaf')
    if s:
        s = 'leaf tables: ' + s
    if only_leaf:
        return s

    global_tables = traverse_TABLE_LIST_low(table_list, 'next_global')
    if global_tables:
        s += '\n'
        s += 'global tables: ' + global_tables
    return s

def print_SELECT_LEX(select_lex):
    """print SELECT_LEX extra information"""

    leaf_tables = select_lex['leaf_tables']
    return traverse_TABLE_LIST(leaf_tables, True)

def print_SELECT_LEX_UNIT(select_lex_unit):
    return ''

# For 8.0.25+
def print_Query_block(query_block):
    """print Query_block extra information"""

    leaf_tables = query_block['leaf_tables']
    return traverse_TABLE_LIST(leaf_tables, True)

def print_Query_expression(unit):
    return ''

class QueryBlockTraverser(gdb.Command, TreeWalker):
    """explore query block tree"""
    def __init__ (self):
        super(self.__class__, self).__init__ ("mysql queryblock", gdb.COMMAND_OBSCURE)

    def invoke(self, arg, from_tty):
        if not arg:
            print("usage: mysql queryblock [SELECT_LEX_UNIT/SELECT_LEX/Query_expression/Query_block]")
            return
        qb = gdb.parse_and_eval(arg)
        self.start_qb = qb.dereference()
        while qb.dereference()['master']:
            qb = qb.dereference()['master']

        self.walk(qb)

    def do_walk_query_block(self, val):
        blocks = []
        if not val['slave']:
            return blocks
        block = val['slave']
        blocks.append(block)
        while block['next']:
            block = block['next']
            blocks.append(block)
        return blocks

    walk_SELECT_LEX = do_walk_query_block
    walk_SELECT_LEX_UNIT = do_walk_query_block
    walk_Query_expression = do_walk_query_block
    walk_Query_block = do_walk_query_block

    def get_current_marker(self, val):
        if self.start_qb.address != val:
            return ''
        return ' <-'

    def show_SELECT_LEX(self, val):
        return print_SELECT_LEX(val) + self.get_current_marker(val)

    def show_SELECT_LEX_UNIT(self, val):
        return print_SELECT_LEX_UNIT(val) + self.get_current_marker(val)

    def show_Query_expression(self, val):
        return print_Query_expression(val) + self.get_current_marker(val)

    def show_Query_block(self, val):
        return print_Query_block(val) + self.get_current_marker(val)

QueryBlockTraverser()

class TABLE_LIST_traverser(gdb.Command):
    """traverse TABLE_LIST list"""
    def __init__ (self):
        super (TABLE_LIST_traverser, self).__init__("mysql tablelist",
                                                    gdb.COMMAND_OBSCURE)
    def invoke(self, arg, from_tty):
        table_list = gdb.parse_and_eval(arg)
        print(traverse_TABLE_LIST(table_list, False))

TABLE_LIST_traverser()

class SEL_TREE_traverser(gdb.Command, TreeWalker, ItemDisplayer):
    """explore SEL_TREE struct"""
    NO_MIN_RANGE = 1
    NO_MAX_RANGE = 2
    NEAR_MIN = 4
    NEAR_MAX = 8
    """print SEL_TREE"""
    def __init__ (self):
        super (self.__class__, self).__init__("mysql seltree", gdb.COMMAND_OBSCURE)

    def invoke(self, arg, from_tty):
        if not arg:
            print("usage: mysql seltree [SEL_TREE]")
            return
        sel_tree = gdb.parse_and_eval(arg)
        if sel_tree:
            self.walk(sel_tree)
        else:
            print('None')

    def walk_SEL_TREE(self, val):
        sel_tree_keys = val['keys']
        return self.sel_tree_keys_to_list(val)

    def show_SEL_TREE(self, val):
        sel_tree_keys = val['keys']
        sel_tree_type = val['type']
        return "[type={},keys.m_size={}]".format(sel_tree_type, sel_tree_keys['m_size'])

    def walk_SEL_ROOT(self, val):
        out_list = []
        if val:
            out_list.append(val['root'])
        return out_list

    def show_SEL_ROOT(self, val):
        if not val:
            return "None"
        sel_root_type = val['type']
        sel_root_use_count = val['use_count']
        sel_root_elements = val['elements']
        return "[type={}, use_count={}, elements={}]".format(sel_root_type, sel_root_use_count, sel_root_elements);

    def walk_SEL_ARG(self, val):
        sel_arg_field = val['field']
        if not sel_arg_field:
             return None
        return self.sel_arg_tree_to_list(val)

    def show_SEL_ARG(self, val):
        sel_arg_field = val['field']
        if not sel_arg_field:
             return None
        level_graph = '  '.join(self.level_graph[:self.current_level])
        for i, c in enumerate(self.level_graph):
            if c == '`':
                self.level_graph[i] = ' '
        left_margin = "  |"

        if len(self.level_graph) < self.current_level + 1:
            self.level_graph.append('|')
        else:
            self.level_graph[self.current_level] = '|'

        field_show_info = ''
        if val['field']:
            field_show_info = self.get_item_show_info(val['field'])
            field_show_info = "{}{} field = {}".format(level_graph, left_margin,
                                    field_show_info)

        sel_root_max_flag = val['max_flag']
        sel_root_min_flag = val['min_flag']
        left_parenthese = '['
        right_parenthese = ']'
        min_item_show_info = ''
        max_item_show_info = ''
        min_value = val['min_value']
        max_value = val['max_value']
        min_item = None
        max_item = None
        try:
            min_item = val['min_item']
            max_item = val['max_item']
        except (gdb.error, RuntimeError):
            min_item = None
            max_item = None

        if min_item and self.NO_MIN_RANGE & sel_root_min_flag == 0:
            min_item_show_info = self.get_item_show_info(val['min_item'])
            if self.NEAR_MIN & sel_root_min_flag > 0:
                left_parenthese = "("
        else:
            if min_item:
                min_item_show_info = " -infinity"
                left_parenthese = "("
            else:
                min_item_show_info = self.get_item_show_info(val['min_value'])
                if self.NEAR_MIN & sel_root_min_flag > 0:
                    left_parenthese = "("

        max_item_show_info = ''
        if max_item and self.NO_MAX_RANGE & sel_root_max_flag == 0:
            max_item_show_info = self.get_item_show_info(val['max_item'])
            if self.NEAR_MAX & sel_root_max_flag > 0:
                right_parenthese = ")"
        else:
            if max_item:
                max_item_show_info = " +infinity"
                right_parenthese = ")"
            else:
                max_item_show_info = self.get_item_show_info(val['max_value'])
                if self.NEAR_MAX & sel_root_max_flag > 0:
                    right_parenthese = ")"

        item_show_info = ''
        if sel_root_max_flag == 0 and sel_root_min_flag == 0 and min_value == max_value:
            item_show_info = "{}{} equal = {}{} {}".format(level_graph, left_margin, left_parenthese, min_item_show_info,
                                                           right_parenthese)
        else:
            item_show_info = "{}{} scope = {}{},{} {}".format(level_graph, left_margin, left_parenthese, min_item_show_info,
                                                              max_item_show_info, right_parenthese)
        return "[color={}, is_asc={}, minflag={}, maxflag={}, part={}]\n{}\n{}".format(
                     val['color'], val['is_ascending'], sel_root_min_flag,
                     sel_root_max_flag, val['part'], field_show_info, item_show_info)

    def get_item_show_info(self, expr):
        item_show_info = ''
        expr_typed = expr.dynamic_type
        expr_casted = expr.cast(expr_typed)
        cname = self.autoncvar.set_var(expr_typed)
        item_show_info = " {} ({}) {}".format(
                         cname, expr_typed, expr)
        show_func = self.get_show_func(expr_typed, None)
        if show_func is not None:
             item_show_info = "{} {}".format(item_show_info, show_func(expr_casted))
        return item_show_info

    def sel_tree_keys_to_list(self, val):
        out_list = []
        sel_tree_keys = val['keys']
        sel_tree_keys_array = sel_tree_keys['m_array']
        for i in range(sel_tree_keys['m_size']):
            out_list.append(sel_tree_keys_array[i])
        return out_list

    def sel_arg_tree_to_list(self, val):
        out_list = []
        sel_arg_left = val['left']
        if sel_arg_left:
            out_list.append(sel_arg_left)
        sel_arg_right = val['right']
        if sel_arg_right:
            out_list.append(sel_arg_right)
        sel_arg_next_part = val['next_key_part']
        if sel_arg_next_part:
            out_list.append(sel_arg_next_part)
        return out_list

SEL_TREE_traverser()

#
# pretty printers
#

def expr_node_value(val, autoncvar):
    """Returns the value held in an list_node<_Val>"""

    val = val.cast(val.dynamic_type)
    return val, autoncvar.set_var(val)

class PrinterIterator(object):
    """A helper class, compatiable with python 2.0"""
    def next(self):
        """For python 2"""
        return self.__next__()

class ListPrinter(object):
    """Print List struct"""

    class _iterator(PrinterIterator):
        def __init__(self, nodetype, head):
            self.nodetype = nodetype
            self.base = head
            self.end_of_list = gdb.parse_and_eval('end_of_list').address
            self.autoncvar = autocvar.AutoNumCVar()

        def __iter__(self):
            return self

        def __next__(self):
            if self.base == self.end_of_list:
                raise StopIteration
            elt = self.base.dereference()
            self.base = elt['next']
            val, cvname = expr_node_value(elt['info'].cast(self.nodetype.pointer()), self.autoncvar)
            return (cvname, '(%s) %s' % (val.dynamic_type, val))

    def __init__(self, val):
        self.typename = val.type
        self.val = val

    def children(self):
        nodetype = self.typename.template_argument(0)
        return self._iterator(nodetype, self.val['first'])

    def to_string(self):
        return '%s' % self.typename if self.val['elements'] != 0 else 'empty %s' % self.typename

# mem_root_deque is from 8.0.22+
class mem_root_dequePrinter(object):
    """Print a MySQL mem_root_deque List"""

    class _iterator(PrinterIterator):
        def __init__(self, nodetype, head):
            self.nodetype = nodetype
            self.base = head['m_begin_idx']
            self.end = head['m_end_idx']
            self.elements = head['block_elements']
            self.blocks = head['m_blocks']
            self.autoncvar = AutoNumCVar()

        def __iter__(self):
            return self

        def __next__(self):
            if self.base == self.end:
                raise StopIteration
            #elt = self.base.dereference()
            elt = self.blocks[self.base / self.elements]['elements'][self.base % self.elements]
            self.base = self.base + 1
            val, cvname = expr_node_value(elt.cast(self.nodetype), self.autoncvar)
            return (cvname, '(%s) %s' % (elt.dynamic_type, val))

    def __init__(self, val):
        self.typename = val.type
        self.val = val

    def children(self):
        nodetype = self.typename.template_argument(0)
        return self._iterator(nodetype, self.val)

    def to_string(self):
        return '%s' % self.typename

import gdb.printing
def build_pretty_printer():
    pp = gdb.printing.RegexpCollectionPrettyPrinter(
        "mysqld")
    pp.add_printer('List', '^List<.*>$', ListPrinter)
    pp.add_printer('mem_root_deque', '^mem_root_deque<.*>$', mem_root_dequePrinter)
    return pp

gdb.printing.register_pretty_printer(
    gdb.current_objfile(),
    build_pretty_printer(),
    True)
