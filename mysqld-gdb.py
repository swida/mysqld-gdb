"""A gdb module for mysql-server development which includes some mysql
commands and some pretty printers.

Please visit https://github.com/swida/mysqld-gdb for latest sources.

Commands:
thread overview -- display thread brief with backtrace aggregation
thread search -- find a thread given a keyword of backtrace function name
mysql digest -- decode a statement digest to readable string
mysql sqlstring -- print running sql statement in current thread
mysql item -- explore expression (Item) tree
mysql queryblock -- explore query block tree
mysql dump_tables -- dump all current query used tables
mysql seltree -- explore SEL_TREE struct
mysql tablelist -- traverse TABLE_LIST list
mysql accesspath -- explore AccessPath tree

Pretty printers of structs:
List
mem_root_deque
mem_root_array
Bounds_checked_array
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

def is_pointer_of(field, typname):
    """check field is a pointer of typname"""
    return field.type.code == gdb.TYPE_CODE_PTR and \
        field.type.target().name == typname

#
# Some convenience variables for debug easily  because they are macros
#
if autocvar.gdb_can_set_cvar:
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

class CurrentRunningSQL(gdb.Command):
    """Print current thread running sql statement"""
    def __init__(self):
        super (CurrentRunningSQL, self).__init__ ("mysql sqlstring", gdb.COMMAND_OBSCURE)

    def print_thd_query_string(self, thd):
        sqlstr = thd['m_query_string']['str']
        print(sqlstr.string()) if sqlstr else print("No sql is running")

    def invoke(self, arg, from_tty):
        sym_thd = gdb.lookup_symbol("current_thd")
        if sym_thd[0] is not None:
            current_thd = gdb.parse_and_eval("current_thd")
            self.print_thd_query_string(current_thd)
            return
        sym_thd = gdb.lookup_symbol("get_current_thd")
        if sym_thd[0] is not None:
            current_thd = gdb.parse_and_eval("get_current_thd()")
            self.print_thd_query_string(current_thd)
        else:
            print("Unknown current thread descriptor.")
CurrentRunningSQL()

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
        # Some gdb could raise a error if it is already the oldest frame
        try:
            frame = frame.older()
        except gdb.MemoryError:
            break
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
        thr_dict = {}
        for thr in threads:
            if not thr.is_stopped():
                continue
            thr.switch()
            bframes = brief_backtrace(None)
            if pattern.search(bframes) is None:
                continue
            if bframes in thr_dict:
                thr_dict[bframes].append(thr.num)
            else:
                thr_dict[bframes] = [thr.num,]
        old_thread.switch()
        thr_ow = [(v,k) for k,v in thr_dict.items()]
        thr_ow.sort(key = lambda l:len(l[0]), reverse=True)
        for nums_thr,funcs in thr_ow:
           gdb.write("[%s] %s\n" % (','.join([str(i) for i in nums_thr]), funcs))
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
            if not thr.is_stopped():
                continue
            thr.switch()
            bframes = brief_backtrace(self.filter_threads)
            if bframes is None:
                continue
            if bframes in thr_dict:
                thr_dict[bframes].append(thr.num)
            else:
                thr_dict[bframes] = [thr.num,]
        old_thread.switch()
        thr_ow = [(v,k) for k,v in thr_dict.items()]
        thr_ow.sort(key = lambda l:len(l[0]), reverse=True)
        for nums_thr,funcs in thr_ow:
           gdb.write("[%s] %s\n" % (','.join([str(i) for i in nums_thr]), funcs))
ThreadOverview()

class TreeWalker(object):
    """A base class for tree traverse"""

    SHOW_FUNC_PREFIX = 'show_'
    WALK_FUNC_PREFIX = 'walk_'

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
        self.current_level = level
        level_graph = '  '.join(self.level_graph[:level])
        for i, c in enumerate(self.level_graph):
            if c == '`':
                self.level_graph[i] = ' '
        cname = self.autoncvar.set_var(expr_casted)
        left_margin = "{}{}".format('' if level == 0 else '--', cname)
        element_show_info = ''
        show_func = self.get_action_func(expr_typed, self.SHOW_FUNC_PREFIX)
        if show_func is not None:
            element_show_info = show_func(expr_casted)
        if element_show_info is not None:
            print("{}{} ({}) {} {}".format(
                  level_graph, left_margin, expr_typed, expr, element_show_info))
        walk_func = self.get_action_func(expr_typed, self.WALK_FUNC_PREFIX)
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
        if element_type.code == gdb.TYPE_CODE_PTR:
            element_type = element_type.target()
        def type_name(typ):
            return typ.name if hasattr(typ, 'name') and typ.name is not None else str(typ)
        func_name = action_prefix + type_name(element_type)
        if hasattr(self, func_name) and callable(getattr(self, func_name)):
            return getattr(self, func_name)

        for field in element_type.fields():
            if not field.is_base_class:
                continue
            typ = field.type
            func_name = action_prefix + type_name(typ)
            if hasattr(self, func_name):
                return getattr(self, func_name)

            return self.get_action_func(typ, action_prefix)

        # Fall through to common action function
        if hasattr(self, action_prefix) and callable(getattr(self, action_prefix)):
            return getattr(self, action_prefix)

        return None

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
    def list_items(self, item_list):
        end_of_list = gdb.parse_and_eval('end_of_list').address
        nodetype = item_list.type.template_argument(0)
        cur_elt = item_list['first']
        children = []
        while cur_elt != end_of_list:
            info = cur_elt.dereference()['info']
            children.append(info.cast(nodetype.pointer()))
            cur_elt = cur_elt.dereference()['next']
        return children

    def walk_Item_equal(self, val):
        return self.list_items(val['fields'])

    def walk_Item_func(self, val):
        children = []
        for i in range(val['arg_count']):
            children.append(val['args'][i])
        return children

    walk_Item_sum = walk_Item_func

    def walk_Item_cond(self, val):
        return self.list_items(val['list'])
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

class QueryBlockTraverser(gdb.Command, TreeWalker):
    """explore query block tree"""
    def __init__ (self):
        super(self.__class__, self).__init__ ("mysql queryblock", gdb.COMMAND_OBSCURE)

    def invoke(self, arg, from_tty):
        if not arg:
            print("usage: mysql queryblock [Query_expression/Query_block]")
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

    walk_Query_expression = do_walk_query_block
    walk_Query_block = do_walk_query_block
    # Support version before 8.0.24
    walk_SELECT_LEX = do_walk_query_block
    walk_SELECT_LEX_UNIT = do_walk_query_block

    def get_current_marker(self, val):
        if self.start_qb.address != val:
            return ''
        return ' <-'

    def show_Query_expression(self, val):
        return self.get_current_marker(val)

    def show_Query_block(self, val):
        select_number = val['select_number']
        leaf_tables = val['leaf_tables']
        table_list = traverse_TABLE_LIST(leaf_tables, True)
        return "#%d, " % select_number + table_list + self.get_current_marker(val)

    # Support version before 8.0.24
    show_SELECT_LEX = show_Query_block
    show_SELECT_LEX_UNIT = show_Query_expression
QueryBlockTraverser()

class DumpAllLeafTables(gdb.Command):
    """Dump current used tables with format of each line: db table1 table2 ..."""
    def __init__(self):
        super(DumpAllLeafTables, self).__init__("mysql dump_tables",
                                            gdb.COMMAND_OBSCURE)
        self.tables = {}

    def add_table(self, db_name, table_name):
        try:
            if table_name not in self.tables[db_name]:
                self.tables[db_name].append(table_name)
        except KeyError:
            self.tables[db_name] = [table_name,]

    def get_leaf_tables(self, table_list):
        TABLE_CATEGORY_USER = 2
        while table_list != 0:
            table_share = table_list['table']['s']
            db_name = table_share['db']['str'].string()
            table_name = table_share['table_name']['str'].string()
            if table_share['table_category'] == TABLE_CATEGORY_USER:
                self.add_table(db_name, table_name)

            table_list = table_list['next_leaf']

    def walk(self, query_expression):
        # top_most must be Query_expression, its slave is a Query_block
        query_block = query_expression['slave']
        while query_block != 0:
            self.get_leaf_tables(query_block['leaf_tables'])
            inner_expression = query_block['slave']
            while inner_expression != 0:
                self.walk(inner_expression)
                inner_expression = inner_expression['next']
            query_block = query_block['next']

    def invoke(self, arg, from_tty):
        if not arg:
            print("usage: mysql dump_tables [Query_expression/Query_block]")
            return
        qb = gdb.parse_and_eval(arg)
        while qb.dereference()['master']:
            qb = qb.dereference()['master']
        self.walk(qb)
        for db, tables in self.tables.items():
            print (db, ' '.join(tables))
DumpAllLeafTables()

class TABLE_LIST_traverser(gdb.Command):
    """traverse TABLE_LIST list"""
    def __init__ (self):
        super (TABLE_LIST_traverser, self).__init__("mysql tablelist",
                                                    gdb.COMMAND_OBSCURE)
    def invoke(self, arg, from_tty):
        table_list = gdb.parse_and_eval(arg)
        print(traverse_TABLE_LIST(table_list, False))
TABLE_LIST_traverser()

class ORDER_traverser(gdb.Command):
    """traverse ORDER list"""
    def __init__(self):
        super (ORDER_traverser, self).__init__("mysql order",
                                               gdb.COMMAND_OBSCURE)
    def invoke(self, arg, from_tty):
        order = gdb.parse_and_eval(arg)
        self.autoncvar = AutoNumCVar()
        ord_typname = order.type.target() if order.type.code == gdb.TYPE_CODE_PTR else order.type.name
        if ord_typname == "ORDER_with_src":
            order = order['order']
        while order:
            v = order.dereference()
            item = v['item']
            direct = v['direction']
            deitem = item.dereference()
            print(self.autoncvar.set_var(v), '{item = ', item, ', *item = ',
                  self.autoncvar.set_var(deitem), '(', deitem.dynamic_type, ') ',
                  deitem, ', direction = ', direct, '}, ', sep='', end='')
            order = v['next']
        print('\b\b')
ORDER_traverser()

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

def find_access_path_struct(access_path):
    """Find actual struct in AccessPath according to AccessPath::type"""
    aptype_name = str(access_path['type']).split('::')[1]
    u = access_path['u']
    aptype_field = None
    for field in u.type.fields():
        if aptype_name.lower() != field.name.lower():
            continue
        aptype_field = field
        break
    else:
        raise "No access path struct found for type: %s" % aptype_name
    return aptype_field

class AccessPathTraverser(gdb.Command, TreeWalker):
    """explore access path tree"""
    def __init__ (self):
        super(self.__class__, self).__init__ ("mysql accesspath", gdb.COMMAND_OBSCURE)

    def invoke(self, arg, from_tty):
        if not arg:
            print("usage: mysql accesspath [accesspath]")
            return
        accesspath = gdb.parse_and_eval(arg)
        self.walk(accesspath)

    def get_materialize_children(self, val):
        query_blocks = val['query_blocks']
        size = int(query_blocks['m_size'])
        array = query_blocks['m_array']
        aps = []
        for i in range(size):
            aps.append(array[i]['subquery_path'])
        return aps

    def walk_AccessPath(self, val):
        apfield = find_access_path_struct(val)
        aptype = val['u'][apfield.name]
        child_aps = []
        for field in aptype.type.fields():
            if is_pointer_of(field, 'AccessPath') and aptype[field.name] != 0:
                child_aps.append(aptype[field.name])
            if is_pointer_of(field, 'MaterializePathParameters'):
                child_aps += self.get_materialize_children(aptype[field])
        return child_aps

    def show_AccessPath(self, val):
        apfield = find_access_path_struct(val)
        aptype = str(val['type']).split('::')[1]
        aptyp_struct = val['u'][apfield.name]
        struct_detail = '{ table = ' + aptyp_struct['table']['alias'].string() + ' }' \
            if aptype == 'TABLE_SCAN' else str(aptyp_struct)
        return aptype + ' ' + \
            self.autoncvar.set_var(aptyp_struct) + ' ' + struct_detail
AccessPathTraverser()

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
            self.autoncvar = AutoNumCVar()

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

class mem_root_arrayPrinter(object):
    """Print List a mem_root_array"""

    class _iterator(PrinterIterator):
        def __init__(self, nodetype, array, size):
            self.nodetype = nodetype
            self.array = array
            self.size = size
            self.index = 0
            self.autoncvar = AutoNumCVar()

        def __iter__(self):
            return self

        def __next__(self):
            if self.index == self.size:
                raise StopIteration
            elt = self.array[self.index]
            self.index += 1
            val, cvname = expr_node_value(elt.cast(self.nodetype), self.autoncvar)
            return (cvname, '(%s) %s' % (val.dynamic_type, val))

    def __init__(self, val):
        self.typename = val.type
        self.val = val

    def children(self):
        nodetype = self.typename.template_argument(0)
        return self._iterator(nodetype, self.val['m_array'], self.val['m_size'])

    def to_string(self):
        return '%s' % self.typename if self.val['m_size'] != 0 else 'empty %s' % self.typename

class Bounds_checked_arrayPrinter(object):
    """Print a Bounds_checked_array"""

    class _iterator(PrinterIterator):
        def __init__(self, nodetype, array, size, ref_item_array):
            self.nodetype = nodetype
            self.array = array
            self.size = size
            self.ref_item_array = ref_item_array
            self.index = 0
            self.autoncvar = AutoNumCVar()

        def __iter__(self):
            return self

        def __next__(self):
            if self.index == self.size:
                raise StopIteration
            elt = self.array[self.index]
            if self.ref_item_array and elt == 0:
                raise StopIteration
            self.index += 1
            val, cvname = expr_node_value(elt.cast(self.nodetype), self.autoncvar)
            if self.ref_item_array:
                return (cvname, '%s: (%s) %s' % (elt.address, val.dynamic_type, val))
            else:
                return (cvname, '(%s) %s' % (val.dynamic_type, val))

    def __init__(self, val):
        self.typename = val.type
        self.val = val

    def children(self):
        nodetype = self.typename.template_argument(0)
        return self._iterator(nodetype, self.val['m_array'], self.val['m_size'], self.typename.name == 'Ref_item_array')

    def to_string(self):
        return '%s' % self.typename if self.val['m_size'] != 0 else 'empty %s' % self.typename

# mem_root_deque is from 8.0.22+
class mem_root_dequePrinter(object):
    """Print a MySQL mem_root_deque List"""

    class _iterator(PrinterIterator):
        def __init__(self, nodetype, head):
            self.nodetype = nodetype
            self.base = head['m_begin_idx']
            self.end = head['m_end_idx']
            self.block_elements = head['block_elements']
            try:
                x = self.base / self.block_elements
            except gdb.error:
                self.block_elements = self.get_block_elements()
            self.blocks = head['m_blocks']
            self.autoncvar = AutoNumCVar()

        def get_block_elements(self):
            """Some gdb version e.g. gdb 9.2 in devtools-10, static
            constexpr block_elements is optimized out, This function
            calculates block_elements dynamically see,
            FindElementsPerBlock() in mem_root_deque.h
            """
            base_number_elems = 1024 / self.nodetype.sizeof
            for block_size in range(16, 1024):
                if block_size >= base_number_elems:
                    return block_size;
                return 1024

        def __iter__(self):
            return self

        def __next__(self):
            if self.base == self.end:
                raise StopIteration
            elt = self.blocks[self.base / self.block_elements]['elements'][self.base % self.block_elements]
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
        return '%s' % self.typename if self.val['m_blocks'] != 0 else 'empty %s' % self.typename

class AccessPathPrinter(object):
    """AccessPath has a big union, this printer make it pretty"""
    def __init__(self, val):
        self.val = val

    def to_string(self):
        s = '{'
        for field in self.val.type.fields():
            if field.name == 'u':
                continue
            if field.name is not None:
                s += field.name + ' = '
                s += str(self.val[field.name]) + ', '
        apfield = find_access_path_struct(self.val)
        s += 'u = {' + apfield.name + ' = ' + str(self.val['u'][apfield.name]) + '}}'
        return s

import gdb.printing
def build_pretty_printer():
    pp = gdb.printing.RegexpCollectionPrettyPrinter(
        "mysqld")
    pp.add_printer('List', '^List<.*>$', ListPrinter)
    pp.add_printer('mem_root_deque', '^mem_root_deque<.*>$', mem_root_dequePrinter)
    pp.add_printer('AccessPath', '^AccessPath$', AccessPathPrinter)
    pp.add_printer('mem_root_array', '^Mem_root_array_YY<.*>$', mem_root_arrayPrinter)
    pp.add_printer('Bounds_checked_array', '^Bounds_checked_array<.*>$', Bounds_checked_arrayPrinter)
    return pp

gdb.printing.register_pretty_printer(
    gdb.current_objfile(),
    build_pretty_printer(),
    True)
