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
import struct
import gdb
#
# A class to generate an unique gdb convenience variable name
#
class AutoCVar(object):
    maxlen = 2
    gdb_can_set_cvar = True
    def __init__(self):
        self.cur_seq = ['a']
        self.gdb_can_set_cvar = hasattr(gdb, 'set_convenience_variable')
        if not self.gdb_can_set_cvar:
            print("Warning: your gdb is too old and does not support set_convenience_variable. You'd better upgrade it.")

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


class ConvenienceVarPrint(gdb.Command):
    """Print convenience variables"""
    def __init__(self):
        super(__class__, self).__init__("cv", gdb.COMMAND_OBSCURE)
    def invoke(self, arg, from_tty):
        cvar_str = arg
        fields = []
        if '.' in cvar_str:
            strs = cvar_str.split('.')
            cvar_str = strs[0]
            fields = strs[1:]
        for v in self.get_all_convenience_vars(cvar_str):
            self.print_cvar(v, fields)

    def print_cvar(self, cv_name, fields):
        val = gdb.convenience_variable(cv_name)
        for field in fields:
            val = val[field]
        if val.type.code == gdb.TYPE_CODE_PTR:
            val = val.dereference()
        prname = '$' + cv_name
        if fields:
            prname += '.' + '.'.join(fields)
        print(prname, '=', val)

    def get_all_convenience_vars(self, cvar_str):
        mvar = re.compile(r"\$" + cvar_str + r"\d+")
        vars = filter(lambda s: s, gdb.execute("show convenience", to_string=True).split('\n'))
        vars = [s.split(' = ')[0] for s in vars]
        vars = [s[1:] for s in vars if mvar.match(s)]
        return vars

ConvenienceVarPrint()

# Define a mysql command prefix for all mysql related command
gdb.Command('mysql', gdb.COMMAND_DATA, prefix=True)

#
# MySQL Commands start here
#

#
# Some small utils
#
class DigestPrinter(gdb.Command):
    """decode a statement digest to readable string"""
    def __init__ (self):
        super (__class__, self).__init__ ("mysql digest", gdb.COMMAND_OBSCURE)

    def invoke(self, arg, from_tty):
        start_addr = gdb.parse_and_eval(arg)
        i = 0
        while i < 32:
            char_code = int(start_addr[i].cast(gdb.lookup_type("unsigned char")))
            print("%02x" % char_code, end='')
            i += 1
        print()
DigestPrinter()

# frame iterators
def foreach_frame(func):
    frame = gdb.newest_frame() if hasattr(gdb, 'newest_frame') else gdb.selected_frame()
    while frame is not None:
        res = func(frame)
        if res:
            break
        # Some gdb could raise a error if it is already the oldest frame
        try:
            frame = frame.older()
        except gdb.MemoryError:
            return

class CurrentRunningSQL(gdb.Command):
    """Print current thread running sql statement"""
    def __init__(self):
        super (CurrentRunningSQL, self).__init__ ("mysql sqlstring", gdb.COMMAND_OBSCURE)

    def print_query_from_local_thd(self, frame):
        try:
            thd = frame.read_var('thd')
            sqlstr = thd['m_query_string']['str']
            if sqlstr:
                print(sqlstr.string())
                return True
        except (gdb.error, gdb.MemoryError, ValueError):
            pass

        return False

    def error(self):
        print("Can not find current thread descriptor.")

    def invoke(self, arg, from_tty):
        # Try to find thread local variable current_thd
        sym_thd = gdb.lookup_symbol("current_thd")
        if sym_thd[0] is not None:
            thd = sym_thd[0].value()
            sqlstr = thd['m_query_string']['str']
            print(sqlstr.string()) if sqlstr else print("No sql is running")
            return
        # if there is no current_thd, Must be reading core, try to find thd
        # local variables in parent stacks
        foreach_frame(self.print_query_from_local_thd)

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

###
### bthread commands
###

"""
bthread stack inspection commands for gdb, core-dump friendly.

Unlike extra/incubator-brpc/incubator-brpc-1.0.0/tools/gdb_bthread_stack.py
(which requires a *live* process because it calls bthread::get_all_bthreads()
and switches $rip/$rsp/$rbp then relies on gdb's built-in `bt`), the
commands here only ever *read* memory, and they unwind manually by walking
the saved frame-pointer (rbp) chain. That makes them work equally well on:
  - a live process (`gdb -p <pid>`)
  - a core file (`gdb <binary> <core>`)

All commands live under the 'bthread' prefix command (run 'help bthread' to
list every subcommand with a one-line summary, or 'help bthread <sub>' for
the full help of one subcommand).

Get Started:
    (gdb) source debug_tools/gdb/bthread_stack.py
    (gdb) bthread ls                 # list all bthreads currently known to
                                      # the bthread runtime (from the
                                      # ResourcePool<TaskMeta> slab allocator)
                                      # -- same as 'info bthreads'
    (gdb) bthread bt <bthread_id>     # print the call stack of one bthread,
                                      # identified by its numeric bthread_id
                                      # (the same id you see in
                                      # information_schema / Vio::bthread_id /
                                      # THD::m_created_bthread_id etc.)
    (gdb) bthread search <keyword>    # list bthreads whose stack contains a
                                      # function name matching keyword
                                      # (case-insensitive substring match)
    (gdb) bthread arg <id> <frame> <arg-index> <type> [rbp-offset]
                                      # recover a spilled argument from a
                                      # suspended bthread frame
    (gdb) bthread switch <bthread_id> # (live process only) borrow the
                                      # current gdb thread to inspect a
                                      # bthread with full native gdb power
                                      # (info locals, frame, p, ...)
    (gdb) bthread restore             # undo 'bthread switch' (usually
                                      # automatic -- see its help)

Notes:
  - A bthread_id can be given in decimal or hex (0x...).
  - If the bthread is currently *actually running* on some pthread (i.e. it
    was the one executing when the core was generated, or is running live),
    its saved ContextualStack does NOT reflect its live registers.
    `bthread bt` detects this automatically (by scanning every LWP's
    thread-local `bthread::tls_task_group->_cur_meta`) and, instead of
    printing a possibly-garbage unwind, tells you which gdb `thread <n>`
    it is running on so you can run `thread <n>` + `bt full` there for the
    real (register-accurate) call stack.
  - If a bthread has no stack (task_meta.stack == 0x0), it has either never
    been scheduled onto a stack yet, or its stack has already been recycled
    into the free stack pool after finishing.
"""
BTHREAD_STACKTYPE_UNKNOWN = 0
BTHREAD_STACKTYPE_PTHREAD = 1

MAX_UNWIND_FRAMES = 64

# --- Why this file never lets gdb "materialize" a BlockGroup/Block value ---
#
# butil::ResourcePool<bthread::TaskMeta>::BlockGroup embeds
# `butil::atomic<Block*> blocks[65536]` -> sizeof(BlockGroup) == 524296
# bytes. Whenever gdb is asked to produce a *value* of this type (e.g. by
# dereferencing a BlockGroup* and then chasing a scalar field such as
# `.nblock`), it fails with a confusing "There is no member named nblock."
# instead of a size-limit error. Root cause, confirmed empirically:
#   - `print *block_group_ptr`            -> works, prints every field.
#   - `block_group_ptr->nblock`           -> "no member named nblock"
#   - Python: bg.dereference()['nblock'] -> same error
#   - Python: bg.dereference().type.fields() -> returns [] (!)
# i.e. gdb's *value* (not type) machinery treats this 512KB struct as an
# "incomplete type" once you try to select a single field out of a fully
# materialized value of it -- and raising `max-value-size` does NOT fix
# this (verified: still fails with `set max-value-size unlimited`), so
# this is a harder internal limit than the max-value-size knob.
#
# The `_resource_pool_task_meta_expr()` lookup below sidesteps this because
# it only ever does pointer arithmetic and address-of-field access (gdb's
# lenient `.`/`->` on a pointer decays an array field straight to its
# address without loading the whole parent struct), never fully
# dereferencing a BlockGroup/Block as a value. That's why it has always
# worked for get_one_bthread()'s O(1) lookup.
#
# For _iter_all_task_metas(), which needs the scalar counters
# BlockGroup::nblock and Block::nitem (to know how many blocks/items are
# live), we avoid the gdb value machinery entirely and read raw bytes off
# the inferior/core memory instead. Layout (confirmed via `ptype /o` and
# cross-checked against a `print *block_group_ptr` dump):
#   BlockGroup { atomic<unsigned long> nblock;      // offset 0, 8 bytes
#                atomic<Block*> blocks[65536]; }     // offset 8
#   Block      { char items[40960];                  // offset 0
#                size_t nitem; }                      // offset 40960, 8 bytes


_BLOCKGROUP_NBLOCK_OFFSET = 0
_BLOCKGROUP_BLOCKS_OFFSET = 8
_BLOCK_NITEM_OFFSET = 40960


def _read_u64(inferior, addr):
    return struct.unpack("<Q", bytes(inferior.read_memory(addr, 8)))[0]


def _resource_pool_task_meta_expr(group, block, item):
    return (
        "*(('bthread::TaskMeta' *)((*(*(('butil::atomic<"
        "butil::ResourcePool<bthread::TaskMeta>::Block*>' *)"
        "((*((*((('butil::static_atomic<"
        "butil::ResourcePool<bthread::TaskMeta>::BlockGroup*>' *)"
        "('butil::ResourcePool<bthread::TaskMeta>::_block_groups')) + {g})).val))"
        ".blocks) + {b}))._M_b._M_p).items) + {i})"
    ).format(g=group, b=block, i=item)


def _block_group_ptr(group):
    """Address of BlockGroup #group (0 if that slot was never allocated)."""
    return int(
        gdb.parse_and_eval(
            "'butil::ResourcePool<bthread::TaskMeta>::_block_groups'[{g}].val".format(
                g=group
            )
        )
    )


def _iter_all_task_metas():
    """Yield every live bthread::TaskMeta found in the ResourcePool slabs.

    Pure memory walk (no inferior function calls, no full-value
    materialization of BlockGroup/Block -- see the note above), safe on a
    core file.
    """
    inferior = gdb.selected_inferior()
    groups = int(
        gdb.parse_and_eval(
            "'butil::ResourcePool<bthread::TaskMeta>::_ngroup'"
        )["val"]
    )
    for group in range(groups):
        bg_ptr = _block_group_ptr(group)
        if bg_ptr == 0:
            continue
        blocks = _read_u64(inferior, bg_ptr + _BLOCKGROUP_NBLOCK_OFFSET)
        for block in range(blocks):
            block_ptr_addr = (
                bg_ptr + _BLOCKGROUP_BLOCKS_OFFSET + block * 8
            )
            block_ptr = _read_u64(inferior, block_ptr_addr)
            if block_ptr == 0:
                continue
            items = _read_u64(inferior, block_ptr + _BLOCK_NITEM_OFFSET)
            for item in range(items):
                task_meta = gdb.parse_and_eval(
                    _resource_pool_task_meta_expr(group, block, item)
                )
                yield task_meta


def _is_valid_task_meta(task_meta):
    version_tid = int(task_meta["tid"]) >> 32
    version_butex = int(gdb.parse_and_eval(
        "*(uint32_t *){}".format(task_meta["version_butex"])
    ))
    return version_tid == version_butex


def get_one_bthread(bthread_id):
    """O(1) lookup of a TaskMeta by bthread_id, using the resource_id encoded
    in the low 32 bits of bthread_id to index directly into the slab."""
    resource_id = bthread_id & 0xFFFFFFFF
    block_index = resource_id // 256
    group_index = block_index >> 16
    offset = resource_id - block_index * 256
    task_meta = gdb.parse_and_eval(
        _resource_pool_task_meta_expr(group_index, block_index, offset)
    )
    if not _is_valid_task_meta(task_meta):
        return None
    return task_meta


def _sym(addr):
    try:
        s = gdb.execute("info symbol 0x{:x}".format(addr), to_string=True).strip()
        return s
    except gdb.error as e:
        return "?? ({})".format(e)


def _source_line(addr):
    """Return 'file:line' for `addr`, or None if no line info is available.

    `addr` should be the address you actually want to attribute (for a
    return address, pass pc-1 so the lookup lands inside the call site's
    statement rather than the instruction after the call).
    """
    try:
        sal = gdb.find_pc_line(addr)
    except gdb.error:
        return None
    if sal is None or sal.symtab is None or sal.line == 0:
        return None
    return "{}:{}".format(sal.symtab.filename, sal.line)


def _func_name(addr):
    """Return just the function name containing `addr` (no offset/section)."""
    fn = _function_range(addr)
    if fn is not None:
        return fn[0]
    return None


def _sym_with_line(addr, is_return_address=False):
    """Resolve `addr` to a human-readable frame description.

    When source line info is available, print a compact
    'function at file:line' (dropping the verbose 'info symbol' tail of
    '+ offset in section .text of <binary>'). Otherwise fall back to the
    full 'info symbol' output so no information is lost.
    """
    # For a return address, the pc points *after* the call, so look up the
    # line/function for pc-1 to attribute it to the call site.
    lookup = addr - 1 if (is_return_address and addr > 0) else addr
    line = _source_line(lookup)
    if line is not None:
        name = _func_name(lookup) or _func_name(addr)
        if name is not None:
            return "{} at {}".format(name, line)
        # Have a line but no clean function name: keep info symbol + line.
        return "{} at {}".format(_sym(addr), line)
    return _sym(addr)


def _bthread_context(task_meta):
    """Return (rip, rbp, rsp) saved in a suspended bthread's ContextualStack.

    Layout matches bthread_make_fcontext's x86_64 save area, consistent
    with extra/incubator-brpc's gdb_bthread_stack.py: the ContextualStack's
    `context` pointer + 6*8/7*8 holds saved rbp/rip, and +8*8 is the
    restored rsp.
    """
    stack = task_meta["stack"]
    if str(stack) == "0x0":
        return None
    context = gdb.parse_and_eval(
        "(*(('bthread::ContextualStack' *){})).context".format(stack)
    )
    rip = int(gdb.parse_and_eval("*(uint64_t*)({}+7*8)".format(context)))
    rbp = int(gdb.parse_and_eval("*(uint64_t*)({}+6*8)".format(context)))
    rsp = int(gdb.parse_and_eval("({}+8*8)".format(context)))
    return rip, rbp, rsp


def _bthread_full_context(task_meta):
    """Return the full callee-saved register set saved in a suspended
    bthread's ContextualStack, as a dict of {reg_name: value}.

    x86-64 save-area layout (confirmed against extra/incubator-brpc's
    gdb_bthread_stack.py BthreadRegsCmd):
        context+1*8 = r12    context+5*8 = rbx
        context+2*8 = r13    context+6*8 = rbp
        context+3*8 = r14    context+7*8 = rip  (return address)
        context+4*8 = r15    rsp = context + 8*8
    The System V AMD64 callee-saved set is {rbx, rbp, r12, r13, r14, r15},
    plus rip/rsp to define the frame -- restoring all of them lets gdb's
    native unwinder compute local variables that live in those registers.
    """
    stack = task_meta["stack"]
    if str(stack) == "0x0":
        return None
    context = gdb.parse_and_eval(
        "(*(('bthread::ContextualStack' *){})).context".format(stack)
    )

    def _u64(off):
        return int(gdb.parse_and_eval("*(uint64_t*)({}+{}*8)".format(context, off)))

    return {
        "r12": _u64(1),
        "r13": _u64(2),
        "r14": _u64(3),
        "r15": _u64(4),
        "rbx": _u64(5),
        "rbp": _u64(6),
        "rip": _u64(7),
        "rsp": int(gdb.parse_and_eval("({}+8*8)".format(context))),
    }


# The bottom of every bthread stack is entered through these functions;
# anything the unwinder produces *above* them is garbage (the fcontext
# stack base has no valid CFI/rbp terminator, so gdb keeps mis-decoding
# leftover bytes as frames). We stop counting valid frames once we've
# passed one of these.
_BTHREAD_STACK_BOTTOM_FUNCS = (
    "bthread_make_fcontext",
    "bthread::TaskGroup::task_runner",
)


def _valid_frame_count(rip, rbp, max_frames=MAX_UNWIND_FRAMES):
    """Count how many frames are genuinely part of this bthread's stack.

    Walk the rbp chain (same logic as `bthread bt`) and count frames whose
    pc resolves to a real function, stopping as soon as we either:
      - hit a pc with no resolvable function (a '?? ()' garbage frame), or
      - just consumed a known stack-bottom function (task_runner /
        bthread_make_fcontext).
    The returned count is suitable as the N in gdb's `bt <N>`, so the
    native backtrace never spills into the bogus frames past the stack
    base.
    """
    frames = _walk_rbp_chain(rip, rbp, max_frames)
    count = 0
    for frame in frames:
        # A 3-tuple means the walk hit an unreadable rbp and stopped.
        if len(frame) == 3:
            break
        _, pc = frame
        if pc is None:
            break
        name = _func_name(pc)
        if name is None:
            # '?? ()' -- past the real stack, stop before counting it.
            break
        count += 1
        if any(bottom in name for bottom in _BTHREAD_STACK_BOTTOM_FUNCS):
            break
    return count if count > 0 else len(frames)


def _running_bthreads():
    """Return {bthread_id: gdb.InferiorThread} for currently running tasks.

    Each worker pthread caches its TaskGroup in the thread-local
    `bthread::tls_task_group`; TaskGroup::_cur_meta identifies the bthread
    currently executing on that pthread. The selected gdb thread is restored
    before returning.
    """
    inferior = gdb.selected_inferior()
    original = gdb.selected_thread()
    running = {}
    try:
        for thr in inferior.threads():
            thr.switch()
            try:
                group = gdb.parse_and_eval("bthread::tls_task_group")
                if int(group) == 0:
                    continue
                cur_meta = group["_cur_meta"]
                if int(cur_meta) == 0:
                    continue
                running[int(cur_meta["tid"])] = thr
            except (gdb.error, RuntimeError, ValueError):
                continue
    finally:
        if original is not None:
            original.switch()
    return running


def _find_running_lwp(bthread_id):
    """Return the pthread/LWP currently executing `bthread_id`, if any."""
    return _running_bthreads().get(bthread_id)


def _walk_thread_stack(thr, max_frames=MAX_UNWIND_FRAMES):
    """Return (frame_no, pc) pairs from a currently running pthread stack.

    Returns [] (never raises) if the frames can't be obtained at all --
    notably gdb.error('Selected thread is running.'), which happens on a
    LIVE attach when this particular thread hasn't actually been stopped
    (e.g. a brand new thread gdb hasn't fully synced yet, or non-stop
    mode). Never happens on a core file (nothing is ever "running"
    there), but callers must not assume a non-empty result either way.
    """
    original = gdb.selected_thread()
    frames = []
    try:
        thr.switch()
        frame = gdb.newest_frame()
    except (gdb.error, RuntimeError):
        frame = None
    try:
        frame_no = 0
        while frame is not None and frame_no < max_frames:
            try:
                frames.append((frame_no, int(frame.pc())))
                frame = frame.older()
                frame_no += 1
            except (gdb.error, RuntimeError):
                break
    finally:
        if original is not None:
            original.switch()
    return frames


def _walk_rbp_chain(rip, rbp, max_frames=MAX_UNWIND_FRAMES):
    """Manually unwind via the classic frame-pointer chain:
       [rbp+0] = saved rbp, [rbp+8] = return address.
    Works identically on a live inferior or a core file since it is pure
    memory reads, no register writes, no target-specific unwinder needed.
    """
    inferior = gdb.selected_inferior()
    frames = [(0, rip)]
    cur_rbp = rbp
    frame_no = 1
    for _ in range(max_frames):
        try:
            mem = inferior.read_memory(cur_rbp, 16)
        except gdb.MemoryError as e:
            frames.append((frame_no, None, "stop: cannot read rbp=0x{:x}: {}".format(cur_rbp, e)))
            break
        saved_rbp, ret_addr = struct.unpack("<QQ", bytes(mem[:16]))
        frames.append((frame_no, ret_addr))
        if saved_rbp == 0 or saved_rbp <= cur_rbp:
            break
        cur_rbp = saved_rbp
        frame_no += 1
    return frames


_ARGUMENT_REGISTERS = (
    ("rdi", "edi", "di", "dil"),
    ("rsi", "esi", "si", "sil"),
    ("rdx", "edx", "dx", "dl"),
    ("rcx", "ecx", "cx", "cl"),
    ("r8", "r8d", "r8w", "r8b"),
    ("r9", "r9d", "r9w", "r9b"),
)


def _saved_frame_at(rip, rbp, target_frame):
    """Return (pc, rbp) for a frame number from `_walk_rbp_chain`."""
    inferior = gdb.selected_inferior()
    pc = rip
    frame_rbp = rbp
    for _ in range(target_frame):
        mem = inferior.read_memory(frame_rbp, 16)
        saved_rbp, return_address = struct.unpack("<QQ", bytes(mem))
        if saved_rbp == 0 or saved_rbp <= frame_rbp:
            return None
        frame_rbp = saved_rbp
        pc = return_address
    return pc, frame_rbp


def _function_range(pc):
    """Return (name, start, end) of the function containing `pc`."""
    block = gdb.block_for_pc(pc)
    while block is not None:
        if block.function is not None:
            return block.function.print_name, int(block.start), int(block.end)
        block = block.superblock
    return None


def _parse_displacement(text):
    if not text:
        return 0
    return int(text, 0)


def _detect_argument_rbp_offset(function_start, function_end, arg_index):
    """Find a canonical x86-64 argument-register spill before the first call."""
    registers = _ARGUMENT_REGISTERS[arg_index]
    end = min(function_end, function_start + 256)
    instructions = gdb.selected_frame().architecture().disassemble(
        function_start, end
    )

    for instruction in instructions:
        asm = instruction["asm"].lower()
        if re.search(r"\bcall\w*\b", asm):
            break
        for register in registers:
            # AT&T: mov %rdi,-0x8(%rbp)
            match = re.search(
                r"\bmov\w*\s+%{}\s*,\s*([+-]?(?:0x[0-9a-f]+|[0-9]+))"
                r"\(%rbp\)".format(register),
                asm,
            )
            if match:
                return _parse_displacement(match.group(1)), asm

            # Intel: mov QWORD PTR [rbp-0x8],rdi
            match = re.search(
                r"\bmov\w*\s+(?:[a-z]+\s+ptr\s+)?"
                r"\[rbp([+-](?:0x[0-9a-f]+|[0-9]+))\]\s*,\s*{}\b".format(
                    register
                ),
                asm,
            )
            if match:
                return _parse_displacement(match.group(1)), asm
    return None


def _lookup_argument_type(type_name):
    """Parse a GDB type expression such as `THD *` without reading memory."""
    if re.match(r"\A[A-Za-z_][A-Za-z0-9_:<>, \t*&\[\]]*\Z", type_name) is None:
        raise ValueError("invalid characters in type")
    return gdb.parse_and_eval("({})0".format(type_name)).type


def _format_rbp_offset(offset):
    if offset < 0:
        return "rbp-0x{:x}".format(-offset)
    if offset > 0:
        return "rbp+0x{:x}".format(offset)
    return "rbp"


class BthreadCmd(gdb.Command):
    """bthread -- prefix command for bthread (fiber) inspection.

    All bthread inspection commands live under this prefix. Run without a
    subcommand (or 'help bthread') to list every subcommand with a short
    summary; run 'help bthread <subcommand>' for one subcommand's full
    documentation.

    SUBCOMMANDS
        bthread ls                    list all known bthreads
        bthread bt <id>                print one bthread's call stack
                                        (read-only, works on core files)
        bthread search <keyword>       find bthreads whose stack hits a
                                        symbol matching <keyword>
        bthread arg <id> <frame> <arg-index> <type> [rbp-offset]
                                        recover a spilled argument
                                        (read-only, works on core files)
        bthread switch <id>             (live process only) borrow the
                                        current gdb thread to inspect a
                                        bthread with full native gdb power
                                        (info locals, frame, p, ...)
        bthread restore                 undo 'bthread switch' (usually
                                        automatic -- see its help)
    """

    def __init__(self):
        gdb.Command.__init__(
            self, "bthread", gdb.COMMAND_STACK, gdb.COMPLETE_NONE, True
        )

    def invoke(self, arg, from_tty):
        # gdb >= 17.1 auto-prints every subcommand's help when a prefix
        # command has no invoke() and is run bare (or with an unmatched
        # sub-command name); older releases just silently do nothing in
        # that case (typing 'bthread' alone prints nothing). Implementing
        # invoke() ourselves and delegating to 'help bthread' gets the
        # same listing on EVERY gdb version. This only ever runs for
        # 'bthread' bare or an unmatched sub-command -- a real
        # sub-command (e.g. 'bthread ls') dispatches straight to its own
        # invoke() and never reaches here.
        gdb.execute("help bthread", from_tty)


class BthreadLsCmd(gdb.Command):
    """bthread ls -- list all known bthreads.

    List every bthread currently tracked by the bthread runtime's
    ResourcePool<TaskMeta> slab allocator.

    ARGUMENTS
        none.

    OUTPUT -- grouped by entry function, most common first
        Bthreads are grouped by their ENTRY function (TaskMeta::fn, the
        worker routine each was created with -- i.e. the bottom of its
        call stack, fixed for its whole lifetime; NOT what it is
        currently executing -- for the current/top frame use
        `bthread bt <id>`, frame #0). Groups are sorted by count,
        largest first. Each group prints as one row:
            <function>[ [<count> bthreads]]   <id>[-<id>][, <id>...]
        - function is printed in FULL (never truncated), left-aligned
          and padded to a fixed-width column when it's short enough to
          fit; a longer (e.g. templated) name simply overflows past
          that column instead of being cut, so the ids that follow are
          pushed right accordingly for that one row only.
        - '[<count> bthreads]' is appended right after the function
          name itself (not a separate column) when count > 1; omitted
          entirely when count == 1 (a single occurrence needs no
          marker).
        - ids is a hex bthread_id list (feed to `bthread bt` /
          `bthread switch`, decimal also works there); consecutive ids
          are collapsed into a single 'a-b' range and always printed on
          one single line, however long that ends up being (no
          wrapping/truncation).

    REQUIREMENTS
        Works on BOTH a live process and a core file (pure memory scan,
        no inferior function calls).

    Also available as 'info bthreads' (same output), for discoverability
    alongside gdb's built-in 'info threads'.

    SEE ALSO
        bthread bt, bthread search, bthread switch, info bthreads
    """

    def __init__(self):
        gdb.Command.__init__(self, "bthread ls", gdb.COMMAND_STACK, gdb.COMPLETE_NONE)

    def invoke(self, arg, tty):
        _bthread_ls_impl()


def _fn_name(fn_val):
    """Human-friendly name for a TaskMeta::fn pointer value: just the
    function name, no leading address and no enclosing '<...>' (unlike
    gdb's default pointer formatting, e.g. '0x55556... <name+0x10>')."""
    addr = int(fn_val)
    name = _func_name(addr)
    if name is not None:
        return name
    # No debug-info-based name available; fall back to whatever gdb's
    # default formatting put inside the '<...>' (still has a symbol name
    # in the common case, just without file:line), or the raw text if
    # even that isn't there.
    text = str(fn_val)
    match = re.search(r"<([^>+]+)", text)
    return match.group(1) if match else text


def _collapse_id_ranges(ids):
    """Collapse a sorted list of ints into ['a', 'b-c', ...] hex strings,
    merging consecutive runs (id[i+1] == id[i]+1) into a single range."""
    ranges = []
    start = prev = ids[0]
    for cur in ids[1:]:
        if cur == prev + 1:
            prev = cur
            continue
        ranges.append((start, prev))
        start = prev = cur
    ranges.append((start, prev))
    return [
        "0x{:x}".format(lo) if lo == hi else "0x{:x}-0x{:x}".format(lo, hi)
        for lo, hi in ranges
    ]


# Layout of a 'bthread ls' row: <function[ [count bthreads]] padded to
# _FUNC_COL_WIDTH>  <ids...>. There is no separate "count" column --
# the count is folded into the function field itself as a
# '[N bthreads]' suffix. Neither the function name nor the ids list is
# ever truncated/wrapped -- a long name just overflows past the
# column (only that row's ids shift right), and all ids are always
# printed on that same single line.
_FUNC_COL_WIDTH = 60


def _format_function_field(fn_name, count):
    """Function name, with a '[N bthreads]' suffix appended when this
    function is the entry point of more than one bthread (count == 1
    gets no suffix -- a lone occurrence needs no marker)."""
    if count > 1:
        return "{} [{} bthreads]".format(fn_name, count)
    return fn_name


def _format_ids(ids):
    """Comma-joined collapsed id ranges, always as a single line (no
    wrapping/truncation, however many ids there are)."""
    return ", ".join(_collapse_id_ranges(sorted(ids)))


def _bthread_ls_impl():
    """Shared listing logic behind 'bthread ls' and 'info bthreads'."""
    groups = {}  # function name -> list of bthread ids
    total = 0
    for task_meta in _iter_all_task_metas():
        if not _is_valid_task_meta(task_meta):
            continue
        if int(task_meta["attr"]["stack_type"]) == BTHREAD_STACKTYPE_UNKNOWN:
            continue
        fn_name = _fn_name(task_meta["fn"])
        groups.setdefault(fn_name, []).append(int(task_meta["tid"]))
        total += 1

    print("{:<{fw}}  {}".format("function", "ids", fw=_FUNC_COL_WIDTH))
    for fn_name, ids in sorted(
        groups.items(), key=lambda kv: (-len(kv[1]), kv[0])
    ):
        fn_field = _format_function_field(fn_name, len(ids))
        print("{:<{fw}}  {}".format(fn_field, _format_ids(ids), fw=_FUNC_COL_WIDTH))
    print("total: {} bthreads, {} distinct functions".format(total, len(groups)))


class InfoBthreadsCmd(gdb.Command):
    """info bthreads -- list all known bthreads (alias of 'bthread ls').

    Same output as 'bthread ls', registered under gdb's built-in 'info'
    prefix so it shows up alongside 'info threads' and friends, and is
    discoverable via 'info' tab-completion / 'help info'.

    ARGUMENTS
        none.

    SEE ALSO
        bthread ls, bthread bt, bthread search, bthread switch
    """

    def __init__(self):
        gdb.Command.__init__(
            self, "info bthreads", gdb.COMMAND_STATUS, gdb.COMPLETE_NONE
        )

    def invoke(self, arg, tty):
        _bthread_ls_impl()


class BthreadBtCmd(gdb.Command):
    """bthread bt <bthread_id> -- print a bthread's call stack (read-only).

    Print the call stack of one suspended bthread by manually walking the
    saved frame-pointer (rbp) chain in its ContextualStack. Each frame is
    shown as 'function at file:line' when debug info is available, else it
    falls back to the raw 'info symbol' output.

    ARGUMENTS
        <bthread_id>   numeric bthread id, decimal or 0x-hex (see
                       'bthread ls' / 'bthread search').

    REQUIREMENTS
        Works on BOTH a live process and a core file. It only READS memory
        and never writes CPU registers, so it is safe to run repeatedly
        and on cores. It does NOT give you 'info locals' / 'p <var>' --
        for that on a LIVE process use 'bthread switch'.

    DECLINED / SPECIAL CASES (an explanation is printed)
        * bthread on a pthread stack   -> find its LWP, 'thread <n>' + bt.
        * bthread currently running     -> its saved context is stale; the
          owning LWP (gdb thread <n>) is reported, use 'thread <n>' + bt.
        * bthread has no stack          -> never scheduled, or recycled.

    SEE ALSO
        bthread switch (live, with locals), bthread ls, bthread search,
        bthread arg
    """

    def __init__(self):
        gdb.Command.__init__(self, "bthread bt", gdb.COMMAND_STACK, gdb.COMPLETE_NONE)

    def invoke(self, arg, tty):
        arg = arg.strip()
        if not arg:
            print("usage: bthread bt <bthread_id>  (decimal or 0x-hex), see 'bthread ls'")
            return
        try:
            bthread_id = int(arg, 0)
        except ValueError:
            print("invalid bthread_id: {}".format(arg))
            return

        task_meta = get_one_bthread(bthread_id)
        if task_meta is None:
            print("bthread_id {} (0x{:x}) not found or stale (version mismatch, "
                  "likely already recycled/reused)".format(bthread_id, bthread_id))
            return

        stack_type = int(task_meta["attr"]["stack_type"])
        if stack_type == BTHREAD_STACKTYPE_PTHREAD:
            print("bthread_id {} runs directly on a pthread (stack_type=PTHREAD); "
                  "find its owning LWP and use 'thread <n>' + 'bt full' instead.".format(bthread_id))
            return

        running_thr = _find_running_lwp(bthread_id)
        if running_thr is not None:
            print(
                "bthread_id {} (0x{:x}) is CURRENTLY RUNNING on LWP {} "
                "(gdb thread {}) -- its saved ContextualStack is stale "
                "(only updated when a bthread is scheduled OUT, not while "
                "it's actually executing).".format(
                    bthread_id, bthread_id, running_thr.ptid[1], running_thr.num
                )
            )
            print(
                "==> Use 'thread {}' then 'bt full' to get the real call "
                "stack instead of this command.".format(running_thr.num)
            )
            return

        ctx = _bthread_context(task_meta)
        if ctx is None:
            print("bthread_id {} has no saved stack (stack==0x0): either never "
                  "scheduled yet, or already finished and its stack was recycled "
                  "back into the free pool.".format(bthread_id))
            return

        rip, rbp, _rsp = ctx
        frames = _walk_rbp_chain(rip, rbp)
        for frame in frames:
            if len(frame) == 3:
                frame_no, _, msg = frame
                print(msg)
                continue
            frame_no, ret_addr = frame
            if ret_addr is None:
                continue
            # frame #0 is the live PC; frames >=1 hold return addresses, so
            # resolve the source line against the call site (pc-1).
            print("#{:<3} 0x{:016x}  {}".format(
                frame_no,
                ret_addr,
                _sym_with_line(ret_addr, is_return_address=(frame_no > 0)),
            ))


class BthreadArgCmd(gdb.Command):
    """bthread arg <bthread_id> <frame_no> <arg_index> <type> [rbp_offset]
    -- recover a spilled function argument from a suspended bthread frame.

    Recover an x86-64 System V argument that was spilled into an
    RBP-relative stack slot, for a frame in a suspended bthread's saved
    stack. Useful when 'bthread switch' + 'info args' is unavailable (e.g.
    on a core file) or the argument was optimized into a stack slot.

    ARGUMENTS
        <bthread_id>   numeric bthread id, decimal or 0x-hex.
        <frame_no>     frame index as printed by 'bthread bt' (0-based).
        <arg_index>    0-based integer/pointer argument position, mapping
                       to the SysV registers: 0=rdi 1=rsi 2=rdx 3=rcx
                       4=r8 5=r9.
        <type>         gdb type of the argument, quoted, e.g. 'THD *'.
                       Only scalar/pointer types up to 8 bytes.
        [rbp_offset]   optional. If omitted, the function prologue is
                       disassembled up to its first call to auto-detect the
                       spill slot. If given (e.g. -8), that rbp-relative
                       slot is read directly.

    RESULT
        Prints the decoded value and also stores it in the convenience
        variable $bthread_arg for further use (e.g. 'p *$bthread_arg').

    REQUIREMENTS
        Works on BOTH a live process and a core file (read-only). Only the
        first 6 integer/pointer args (register args) are recoverable, and
        only if they were spilled to the stack.

    EXAMPLE
        (gdb) bthread arg 4294973925 28 0 'THD *'
        (gdb) bthread arg 4294973925 28 0 'THD *' -8      # manual offset

    SEE ALSO
        bthread bt, bthread switch (live: 'info args' is easier)
    """

    def __init__(self):
        gdb.Command.__init__(
            self, "bthread arg", gdb.COMMAND_STACK, gdb.COMPLETE_NONE
        )

    def invoke(self, arg, tty):
        try:
            args = gdb.string_to_argv(arg)
        except gdb.error as error:
            print("invalid arguments: {}".format(error))
            return

        if len(args) not in (4, 5):
            print(
                "usage: bthread arg <bthread_id> <frame_no> <arg_index> "
                "<type> [rbp_offset]"
            )
            print("example: bthread arg 4294973925 28 0 'THD *'")
            print("manual:  bthread arg 4294973925 28 0 'THD *' -8")
            return

        try:
            bthread_id = int(args[0], 0)
            frame_no = int(args[1], 0)
            arg_index = int(args[2], 0)
            explicit_offset = int(args[4], 0) if len(args) == 5 else None
        except ValueError as error:
            print("invalid numeric argument: {}".format(error))
            return

        if frame_no < 0 or frame_no > MAX_UNWIND_FRAMES:
            print("frame_no must be between 0 and {}".format(MAX_UNWIND_FRAMES))
            return
        if arg_index < 0 or arg_index >= len(_ARGUMENT_REGISTERS):
            print("arg_index must be between 0 and 5 on x86-64 SysV")
            return

        type_name = args[3]
        try:
            argument_type = _lookup_argument_type(type_name)
            type_size = int(argument_type.sizeof)
        except (gdb.error, RuntimeError, TypeError, ValueError) as error:
            print("invalid or incomplete type {!r}: {}".format(type_name, error))
            return
        if type_size <= 0 or type_size > 8:
            print(
                "unsupported argument type size {}: only scalar and pointer "
                "values up to 8 bytes are supported".format(type_size)
            )
            return

        task_meta = get_one_bthread(bthread_id)
        if task_meta is None:
            print(
                "bthread_id {} (0x{:x}) not found or stale".format(
                    bthread_id, bthread_id
                )
            )
            return

        stack_type = int(task_meta["attr"]["stack_type"])
        if stack_type == BTHREAD_STACKTYPE_PTHREAD:
            print("bthread_id {} uses a pthread stack; use native gdb frames".format(
                bthread_id
            ))
            return

        running_thr = _find_running_lwp(bthread_id)
        if running_thr is not None:
            print(
                "bthread_id {} is currently running on gdb thread {}; its "
                "saved stack is stale. Use 'thread {}', 'frame {}', and "
                "'info args' instead.".format(
                    bthread_id, running_thr.num, running_thr.num, frame_no
                )
            )
            return

        ctx = _bthread_context(task_meta)
        if ctx is None:
            print("bthread_id {} has no saved stack".format(bthread_id))
            return

        rip, rbp, _ = ctx
        try:
            frame = _saved_frame_at(rip, rbp, frame_no)
        except (gdb.error, gdb.MemoryError, RuntimeError, ValueError) as error:
            print("cannot unwind to frame #{}: {}".format(frame_no, error))
            return
        if frame is None:
            print("frame #{} is not present in the saved RBP chain".format(frame_no))
            return

        pc, frame_rbp = frame
        function = _function_range(pc)
        if function is None:
            function_name = _sym(pc)
            function_start = None
            function_end = None
        else:
            function_name, function_start, function_end = function

        detected_instruction = None
        rbp_offset = explicit_offset
        if rbp_offset is None:
            if function_start is None:
                print(
                    "cannot determine the function containing frame #{}; "
                    "specify rbp_offset explicitly".format(frame_no)
                )
                return
            try:
                detected = _detect_argument_rbp_offset(
                    function_start, function_end, arg_index
                )
            except (gdb.error, RuntimeError, ValueError) as error:
                print("cannot disassemble {}: {}".format(function_name, error))
                return
            if detected is None:
                print(
                    "cannot find a spill of argument #{} ({}) before the first "
                    "call in {}; specify rbp_offset explicitly".format(
                        arg_index,
                        _ARGUMENT_REGISTERS[arg_index][0],
                        function_name,
                    )
                )
                return
            rbp_offset, detected_instruction = detected

        slot_address = frame_rbp + rbp_offset
        try:
            raw_bytes = bytes(
                gdb.selected_inferior().read_memory(slot_address, type_size)
            )
            raw_value = int.from_bytes(raw_bytes, byteorder="little", signed=False)
            value = gdb.Value(raw_value).cast(argument_type)
            gdb.set_convenience_variable("bthread_arg", value)
        except (gdb.error, gdb.MemoryError, RuntimeError, ValueError) as error:
            print("cannot read argument at 0x{:x}: {}".format(slot_address, error))
            return

        print("bthread_id: {} (0x{:x})".format(bthread_id, bthread_id))
        print("frame:      #{} {}".format(frame_no, function_name))
        print("pc:         0x{:016x}".format(pc))
        print("rbp:        0x{:016x}".format(frame_rbp))
        print(
            "argument:   #{} ({})".format(
                arg_index, _ARGUMENT_REGISTERS[arg_index][0]
            )
        )
        print(
            "storage:    [{}] = 0x{:016x}".format(
                _format_rbp_offset(rbp_offset), slot_address
            )
        )
        if detected_instruction is not None:
            print("detected:   {}".format(detected_instruction))
        print("value:      ({}) {}".format(type_name, value))
        print("saved as:   $bthread_arg")


class BthreadSearchCmd(gdb.Command):
    """bthread search <keyword> -- find bthreads whose stack hits a symbol.

    Scan every known bthread and list those whose call stack contains a
    frame with a function name matching <keyword> (case-insensitive
    substring match). Hits are grouped by their matched frame -- same
    '#<frame_no> function at file:line' text (no raw address; falls back
    to 'info symbol' output when no debug info is available) -- and
    printed the same way as `bthread ls`, most common first:
        <matched frame>[ [<count> bthreads]]   <id>[-<id>][, <id>...]
    See `bthread ls`'s help for the exact column/wrap/suffix rules
    (they're shared with this command).

    ARGUMENTS
        <keyword>   substring to look for in frame function names, e.g.
                    'RowExchange' or 'butex_wait'.

    HOW STACKS ARE OBTAINED
        * suspended bthread   -> unwound from its saved ContextualStack.
        * currently running    -> unwound from the real pthread stack of the
          LWP it runs on; on a LIVE process, if that particular LWP
          hasn't actually been stopped yet (gdb reports "Selected thread
          is running"), it is skipped instead of erroring out -- counted
          under 'skipped'.
        * pthread-stack bthread that is not running -> skipped (no usable
          saved stack); counted under 'skipped'.

    REQUIREMENTS
        Works on BOTH a live process and a core file (read-only); never
        raises even if a live LWP can't be unwound (see above).

    SEE ALSO
        bthread ls, bthread bt, bthread switch
    """

    def __init__(self):
        gdb.Command.__init__(
            self, "bthread search", gdb.COMMAND_STACK, gdb.COMPLETE_NONE
        )

    def invoke(self, arg, tty):
        keyword = arg.strip()
        if not keyword:
            print("usage: bthread search <keyword>")
            return

        keyword_lower = keyword.lower()
        running = _running_bthreads()
        symbol_cache = {}
        searched = 0
        skipped = 0
        groups = {}  # matched frame description -> list of bthread ids

        for task_meta in _iter_all_task_metas():
            if not _is_valid_task_meta(task_meta):
                continue

            stack_type = int(task_meta["attr"]["stack_type"])
            if stack_type == BTHREAD_STACKTYPE_UNKNOWN:
                continue

            tid = int(task_meta["tid"])
            searched += 1
            thr = running.get(tid)

            if thr is not None:
                # See _walk_thread_stack: on a live attach this can come
                # back [] if gdb reports "Selected thread is running"
                # for this particular thread (not stopped yet) -- treat
                # that the same as any other unusable stack.
                frames = _walk_thread_stack(thr)
                if not frames:
                    skipped += 1
                    continue
            elif stack_type == BTHREAD_STACKTYPE_PTHREAD:
                skipped += 1
                continue
            else:
                try:
                    ctx = _bthread_context(task_meta)
                    if ctx is None:
                        skipped += 1
                        continue
                    rip, rbp, _ = ctx
                    frames = _walk_rbp_chain(rip, rbp)
                except (gdb.error, gdb.MemoryError, RuntimeError, ValueError):
                    skipped += 1
                    continue

            matched_frame = None
            for frame in frames:
                if len(frame) != 2:
                    continue
                frame_no, pc = frame
                if pc is None:
                    continue
                symbol = symbol_cache.get(pc)
                if symbol is None:
                    symbol = _sym(pc)
                    symbol_cache[pc] = symbol
                if keyword_lower in symbol.lower():
                    matched_frame = (frame_no, pc, symbol)
                    break

            if matched_frame is None:
                continue

            frame_no, pc, _symbol = matched_frame
            desc = "#{:<3} {}".format(
                frame_no, _sym_with_line(pc, is_return_address=(frame_no > 0))
            )
            groups.setdefault(desc, []).append(tid)

        print("searching bthread stacks for {!r} ...".format(keyword))
        print("{:<{fw}}  {}".format("matched frame", "ids", fw=_FUNC_COL_WIDTH))
        matched = 0
        for desc, ids in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            matched += len(ids)
            field = _format_function_field(desc, len(ids))
            print("{:<{fw}}  {}".format(field, _format_ids(ids), fw=_FUNC_COL_WIDTH))

        print(
            "matched: {}, searched: {}, skipped(no usable stack): {}, "
            "distinct frames: {}".format(matched, searched, skipped, len(groups))
        )


# Saved real register context of switched gdb thread(s), so that
# `bthread switch` can be undone by `bthread restore`. Keyed by the gdb
# global thread id (InferiorThread.global_num) it was taken on.
#
# INVARIANT: at most ONE entry lives here at any time. gdb's 'backtrace
# limit' (used to hide the garbage frames past a bthread stack's base --
# see _valid_frame_count) is a *global* gdb setting, not per-thread, so
# it can only be correct for a single switched thread at once. Letting
# two different gdb threads stay switched simultaneously would make the
# limit silently wrong (truncated or garbage-showing) for whichever one
# isn't the most recently switched. `bthread switch` enforces this by
# auto-restoring any *other* thread's registers before switching a new
# one; re-switching the SAME thread to a different bthread is exempt
# (see the setdefault() below) since it doesn't add a second entry.
_saved_thread_regs = {}

# The global 'backtrace limit' gdb setting as it was before the first
# active 'bthread switch', so it can be restored once no switched thread
# remains. (gdb's backtrace limit is a global setting, not per-thread.)
_original_backtrace_limit = None

# Full System V AMD64 callee-saved set plus rip/rsp. Saving/restoring all
# of these (not just rip/rbp/rsp) lets gdb's native unwinder correctly
# recover locals held in rbx/r12-r15 in the switched-to bthread frames.
_SWITCH_REGISTERS = ("rip", "rsp", "rbp", "rbx", "r12", "r13", "r14", "r15")


def _read_registers(regs):
    frame = gdb.selected_frame()
    return {name: int(frame.read_register(name)) for name in regs}


def _write_registers(values):
    for name, value in values.items():
        gdb.execute("set $%s = 0x%x" % (name, value & 0xFFFFFFFFFFFFFFFF),
                    to_string=True)


def _get_backtrace_limit():
    """Return gdb's current 'backtrace limit' (0 means unlimited)."""
    try:
        out = gdb.execute("show backtrace limit", to_string=True)
    except gdb.error:
        return 0
    match = re.search(r"(\d+)", out)
    return int(match.group(1)) if match else 0


def _set_backtrace_limit(n):
    gdb.execute("set backtrace limit {}".format(n), to_string=True)


def _restore_saved_registers(key):
    """Write back one gdb thread's real registers, saved earlier by
    'bthread switch' under gdb global thread id `key`.

    Returns (thr_num, True) on success, (thr_num, False) if the write
    failed or the thread no longer exists (the saved entry is put back
    in the False case so it can be retried), or None if `key` was not
    present in `_saved_thread_regs` at all.
    """
    entry = _saved_thread_regs.pop(key, None)
    if entry is None:
        return None
    thr_num, saved = entry
    original = gdb.selected_thread()
    try:
        target_thr = next(
            t for t in gdb.selected_inferior().threads()
            if t.global_num == key
        )
    except StopIteration:
        # Thread no longer exists (e.g. it exited); nothing to write.
        return (thr_num, False)
    try:
        target_thr.switch()
        _write_registers(saved)
        gdb.execute("flushregs", to_string=True)
    except gdb.error as e:
        _saved_thread_regs[key] = entry
        print("bthread_stack: failed to auto-restore gdb thread {}: {}"
              .format(thr_num, e))
        return (thr_num, False)
    finally:
        if original is not None and original.is_valid():
            original.switch()
    return (thr_num, True)


def _auto_restore_all(reason, keep_key=None):
    """Restore every switched gdb thread except `keep_key` (if given).

    This is the shared engine behind making 'bthread restore' OPTIONAL.
    It is invoked automatically from three places:
      - 'bthread switch' itself, before switching a different gdb thread
        (keep_key = the thread about to be switched).
      - the 'selected_context' event listener below, when the user
        navigates away from a switched thread via 'thread <n>' (or any
        other means gdb uses to change the selected thread).
      - the hook-continue/step/next/.../advance CLI hooks below, just
        before the inferior is actually resumed.
    """
    stale = [k for k in _saved_thread_regs if k != keep_key]
    if not stale:
        return
    restored = []
    for key in stale:
        result = _restore_saved_registers(key)
        if result is not None and result[1]:
            restored.append(result[0])
    if restored:
        print("bthread_stack: auto-restored gdb thread(s) {} {}.".format(
            ", ".join(str(n) for n in restored), reason))
    if not _saved_thread_regs:
        global _original_backtrace_limit
        if _original_backtrace_limit is not None:
            _set_backtrace_limit(_original_backtrace_limit)
            _original_backtrace_limit = None


def _on_selected_context(event):
    """gdb.events.selected_context listener: fires whenever the user
    changes gdb's selected inferior/thread/frame (via 'thread', 'frame',
    'inferior', or equivalent). If that leaves some OTHER gdb thread
    still in the 'bthread switch'-ed state, auto-restore it -- so simply
    moving on with 'thread <n>' is enough, no manual 'bthread restore'
    needed.

    Only available on gdb releases new enough to have
    'gdb.events.selected_context' (added upstream in 2026; most
    currently-deployed gdb versions do NOT have it yet). See
    '_on_before_prompt' below for the fallback used otherwise.
    """
    new_thread = getattr(event, "thread", None)
    new_key = new_thread.global_num if new_thread is not None else None
    _auto_restore_all("because you switched away from it", keep_key=new_key)


# Tracks the gdb global thread id selected the last time we checked, for
# the '_on_before_prompt' polling fallback below. Only meaningful when
# 'gdb.events.selected_context' doesn't exist.
_last_selected_thread_key = "__unset__"


def _on_before_prompt():
    """Fallback for gdb releases without 'gdb.events.selected_context'
    (i.e. most currently-deployed versions -- that event was only added
    upstream in 2026). 'before_prompt' fires right before gdb redisplays
    its prompt, which happens after every command including 'thread
    <n>', so polling the selected thread here and comparing against what
    it was last time we checked detects the same "switched away from a
    switched thread" case, just one command later than the real event
    would.
    """
    global _last_selected_thread_key
    if not _saved_thread_regs:
        # Nothing switched, no point polling; also keeps this cheap.
        _last_selected_thread_key = "__unset__"
        return
    try:
        thr = gdb.selected_thread()
    except gdb.error:
        return
    new_key = thr.global_num if (thr is not None and thr.is_valid()) else None
    if new_key == _last_selected_thread_key:
        return
    _last_selected_thread_key = new_key
    _auto_restore_all("because you switched away from it", keep_key=new_key)


def _bthread_auto_restore_hook(command_name):
    """Invoked (via the CLI 'hook-<command>' definitions at the bottom of
    this file) right BEFORE gdb actually resumes the inferior for
    'continue'/'step'/'next'/etc. This is the only safety net that runs
    *before* resumption: gdb's Python 'events.cont' fires only *after*
    the target has already resumed, too late to prevent a thread from
    running with a bthread's borrowed registers.
    """
    _auto_restore_all("before '{}' resumes it".format(command_name))


class BthreadSwitchCmd(gdb.Command):
    """bthread switch <bthread_id> -- borrow a pthread to debug a bthread.

    Repoint the CURRENTLY SELECTED gdb thread's CPU registers at a
    suspended bthread's saved context, so that all native gdb inspection
    commands operate on that bthread's stack, right away and with no
    extra commands to remember. After switching you can use:
        bt / bt full        backtrace, auto-capped to the real frames --
                             the fcontext stack base has no unwind
                             terminator, so this command lowers gdb's
                             'backtrace limit' for you to hide the '?? ()'
                             garbage frames past the bottom.
        frame <N>           select a frame
        info locals         local variables (incl. those in rbx/r12-r15)
        info args           function arguments
        p <expr> / ptype    evaluate expressions in the frame's scope
    You do NOT need to remember to run 'bthread restore' -- switching
    away, resuming, or switching to a different bthread all restore the
    borrowed registers for you automatically (see AUTOMATIC RESTORE
    below). Run 'bthread restore' manually only if you want the real
    registers back immediately without doing one of those things.

    ARGUMENTS
        <bthread_id>   numeric bthread id, decimal or 0x-hex. Get it from
                       'bthread ls', 'bthread search', or the id shown in
                       information_schema / THD::m_created_bthread_id.

    WHAT IS SWITCHED
        The full System V AMD64 callee-saved set plus the frame regs:
            rip, rsp, rbp, rbx, r12, r13, r14, r15
        (read from the bthread's ContextualStack save area). Switching all
        of them -- not just rip/rbp/rsp -- lets gdb recover locals that are
        held in rbx/r12-r15, so 'info locals' is accurate.

    REQUIREMENTS
        LIVE PROCESS ONLY. This WRITES CPU registers, which a core file
        does not support (use 'bthread bt' for a read-only unwind on
        cores). Stop execution first (all threads stopped is fine).

    AUTOMATIC RESTORE -- 'bthread restore' is now mostly OPTIONAL
        Because gdb's 'backtrace limit' is a GLOBAL setting (not
        per-thread), only ONE gdb thread is ever left switched at a time,
        and its real registers are put back for you automatically when:
          * You run 'bthread switch' again, on this or another gdb
            thread -- the previously switched thread is auto-restored
            first.
          * You change gdb's selected thread ('thread <n>') -- caught via
            gdb's 'selected_context' event on gdb releases that have it
            (added upstream in 2026), or by polling on 'before_prompt'
            (one command later) on older releases that don't.
          * You resume the inferior at all ('continue'/'step'/'next'/
            'finish'/'until'/'advance'/'run') -- caught via 'hook-<cmd>',
            which runs *before* gdb actually resumes the target (unlike
            the Python 'events.cont' notification, which only fires
            *after* resuming -- too late to prevent it), so the CPU is
            guaranteed to never actually execute with borrowed registers.
        These three hooks together mean plain 'thread <n>' or 'continue'
        is enough -- you never HAVE to type 'bthread restore'.

    SAFETY -- READ THIS
        * The automatic restore above covers the standard CLI commands
          and the ordinary 'thread'/'frame' navigation. It does NOT
          cover every conceivable way to resume the target -- e.g.
          calling gdb.execute("continue") from your OWN Python/MI code,
          or a command alias that bypasses hook-<realname>, will NOT be
          intercepted. When in doubt, run 'bthread restore' explicitly.
        * The original registers remain saved (and 'info locals' etc.
          keep working) until one of the restore triggers above fires,
          or you run 'bthread restore' yourself.

    DECLINED CASES (nothing is switched, an explanation is printed)
        * bthread runs on a pthread stack   -> just 'thread <n>' its LWP.
        * bthread is currently running       -> its saved context is stale;
          'thread <n>' to the owning LWP instead (see 'bthread bt').
        * bthread has no stack               -> never scheduled, or already
          finished and recycled.

    EXAMPLE
        (gdb) bthread switch 34359748649
        (gdb) bt
        (gdb) frame 14
        (gdb) info locals
        (gdb) bthread restore

    SEE ALSO
        bthread restore, bthread bt, bthread ls, bthread search, bthread arg
    """

    def __init__(self):
        gdb.Command.__init__(
            self, "bthread switch", gdb.COMMAND_STACK, gdb.COMPLETE_NONE
        )

    def invoke(self, arg, tty):
        arg = arg.strip()
        if not arg:
            print("usage: bthread switch <bthread_id>  (decimal or 0x-hex), "
                  "see 'bthread ls'")
            return
        try:
            bthread_id = int(arg, 0)
        except ValueError:
            print("invalid bthread_id: {}".format(arg))
            return

        thr = gdb.selected_thread()
        if thr is None or not thr.is_valid():
            print("no selected thread; run this on a live, stopped process")
            return

        # Verify we can write registers (fails cleanly on a core file).
        try:
            _ = gdb.selected_frame().read_register("rip")
        except gdb.error as e:
            print("cannot read registers ({}). This command needs a live, "
                  "stopped process (not a core file).".format(e))
            return

        task_meta = get_one_bthread(bthread_id)
        if task_meta is None:
            print("bthread_id {} (0x{:x}) not found or stale".format(
                bthread_id, bthread_id))
            return

        stack_type = int(task_meta["attr"]["stack_type"])
        if stack_type == BTHREAD_STACKTYPE_PTHREAD:
            print("bthread_id {} runs directly on a pthread; just select its "
                  "LWP with 'thread <n>' -- no switch needed.".format(bthread_id))
            return

        running_thr = _find_running_lwp(bthread_id)
        if running_thr is not None:
            print("bthread_id {} is CURRENTLY RUNNING on gdb thread {}; its "
                  "saved context is stale. Use 'thread {}' directly.".format(
                      bthread_id, running_thr.num, running_thr.num))
            return

        ctx = _bthread_full_context(task_meta)
        if ctx is None:
            print("bthread_id {} has no saved stack (never scheduled or "
                  "already recycled)".format(bthread_id))
            return

        key = thr.global_num

        # 'backtrace limit' is a *global* gdb setting (not per-thread), so
        # at most one gdb thread may be left in the switched state -- see
        # the _saved_thread_regs comment above. If some OTHER thread is
        # still switched, auto-restore it first (shared with the
        # selected_context/hook-continue safety nets below).
        _auto_restore_all(
            "because gdb thread {} is switching to a different bthread".format(
                thr.num),
            keep_key=key,
        )

        if key in _saved_thread_regs:
            print("gdb thread {} already switched to a bthread; switching "
                  "it to a different bthread now (original registers stay "
                  "saved for 'bthread restore').".format(thr.num))

        try:
            saved = _read_registers(_SWITCH_REGISTERS)
            _saved_thread_regs.setdefault(key, (thr.num, saved))
            _write_registers(ctx)
        except gdb.error as e:
            print("failed to switch registers: {}".format(e))
            return

        gdb.execute("flushregs", to_string=True)

        # The fcontext stack base has no valid unwind terminator, so a bare
        # 'bt' would spill hundreds of bogus '?? ()' frames past the real
        # bottom. Cap gdb's global 'backtrace limit' to the genuine frame
        # count, so plain 'bt'/'bt full'/'where' just work without needing
        # a special command or an explicit count.
        global _original_backtrace_limit
        n = _valid_frame_count(ctx["rip"], ctx["rbp"])
        if _original_backtrace_limit is None:
            _original_backtrace_limit = _get_backtrace_limit()
        _set_backtrace_limit(n)

        print("gdb thread {}: registers pointed at bthread {} (0x{:x}).".format(
            thr.num, bthread_id, bthread_id))
        print("switched: rip/rsp/rbp/rbx/r12-r15 (full callee-saved set).")
        print("backtrace limit capped to {} real frames (stack-base garbage "
              "past that is hidden). 'bthread restore' undoes this too."
              .format(n))
        print("")
        gdb.execute("bt", to_string=False)


class BthreadRestoreCmd(gdb.Command):
    """bthread restore -- manually undo a 'bthread switch' (usually optional).

    Restore the currently selected gdb thread's original CPU registers
    (rip, rsp, rbp, rbx, r12-r15) that were saved by 'bthread switch',
    returning the thread to its real execution context. Once no gdb
    thread is left switched, also restores gdb's global 'backtrace limit'
    to whatever it was before the first 'bthread switch'.

    YOU USUALLY DON'T NEED TO RUN THIS -- IT'S AUTOMATIC
        The same restore logic now fires on its own in the situations
        that used to require calling this by hand:
          * Switching a DIFFERENT bthread ('bthread switch <id>' again,
            same or another gdb thread) auto-restores whichever thread
            was previously switched.
          * Selecting a different gdb thread ('thread <n>') auto-restores
            the thread you're leaving, via gdb's selected_context event.
          * Resuming execution ('continue'/'step'/'next'/'finish'/'until'/
            'advance'/'run') auto-restores EVERY still-switched thread
            first, via 'hook-<command>', so the CPU never actually runs
            with a borrowed bthread register set.
        Manually running 'bthread restore' is now mainly useful if you
        want to go back to the real registers immediately, without
        performing one of the actions above (e.g. right before ending
        the gdb session, or just to double check).

    ARGUMENTS
        none. Operates on the currently selected gdb thread. Select the
        same thread you ran 'bthread switch' on before restoring.

    NOTES
        * If the selected thread was never switched, prints a notice and
          does nothing.

    SEE ALSO
        bthread switch
    """

    def __init__(self):
        gdb.Command.__init__(
            self, "bthread restore", gdb.COMMAND_STACK, gdb.COMPLETE_NONE
        )

    def invoke(self, arg, tty):
        thr = gdb.selected_thread()
        if thr is None or not thr.is_valid():
            print("no selected thread")
            return
        result = _restore_saved_registers(thr.global_num)
        if result is None:
            print("gdb thread {} was not switched (nothing to restore)".format(
                thr.num))
            return
        thr_num, ok = result
        if not ok:
            return  # error already printed by _restore_saved_registers

        # Once no gdb thread is left switched, restore the original global
        # backtrace limit that 'bthread switch' overrode.
        if not _saved_thread_regs:
            global _original_backtrace_limit
            if _original_backtrace_limit is not None:
                _set_backtrace_limit(_original_backtrace_limit)
                _original_backtrace_limit = None

        print("gdb thread {}: original registers restored.".format(thr_num))


BthreadCmd()
BthreadLsCmd()
BthreadBtCmd()
BthreadArgCmd()
BthreadSearchCmd()
BthreadSwitchCmd()
BthreadRestoreCmd()
InfoBthreadsCmd()

# --- Automatic safety nets, so 'bthread restore' is optional in practice ---
#
# 1) gdb.events.selected_context fires synchronously right after the user
#    changes the selected inferior/thread/frame (e.g. 'thread <n>'). This
#    event was only added upstream in 2026, so most currently-deployed
#    gdb builds don't have it yet ('_gdbevents' has no attribute
#    'selected_context') -- fall back to polling on 'before_prompt',
#    which exists in far older gdb releases, instead.
if hasattr(gdb.events, "selected_context"):
    gdb.events.selected_context.connect(_on_selected_context)
else:
    gdb.events.before_prompt.connect(_on_before_prompt)

# 2) hook-<command> is gdb's pre-execution hook mechanism (CLI 'define
#    hook-continue' etc.): unlike the Python 'events.cont' notification
#    (which only fires *after* the inferior has already resumed, too
#    late), these run *before* gdb actually resumes the target, so this
#    is the only place we can restore borrowed registers in time to stop
#    the CPU from ever running with them.
for _hooked_cmd in ("continue", "step", "next", "finish", "until", "advance", "run"):
    gdb.execute(
        "define hook-{0}\n"
        "python _bthread_auto_restore_hook({0!r})\n"
        "end".format(_hooked_cmd),
        to_string=True,
    )
del _hooked_cmd

###
### MySQL structure explore tools.
###
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
            typ_name = typ.name if hasattr(typ, 'name') and typ.name is not None else str(typ)
            return typ_name.replace('::', '_')
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
    def show_Item_field(self, item):
        field = item['field']
        if not field or not field['field_name']:
            return self.show_Item_ident(item)
        field_info = []
        if field['table']['alias']:
            field_info.append(field['table']['alias'].string())
        else:
            field_info.append(field['table']['s']['table_name'].string())
        field_info.append(field['field_name'].string())
        return 'field = ' + '.'.join(field_info)

    def show_Item_int(self, item):
        return 'value = ' + str(item['value'])

    show_Item_float = show_Item_int

    def show_Item_string(self, item):
        ptr = item['str_value']['m_ptr']
        return 'value = ' + (ptr.string() if ptr else '')

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

    def walk_Item_cache(self, val):
        return [val['example'],]
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

# Traverse Query expression / Query block tree, output like this:
# $nv0 (Query_expression *) 0x7ffff1e81858
# `--$nv1 (Query_block *) 0x7ffff1e81fc8 #1, leaf tables: ($nw0)t2 <-
#    `--$nv2 (Query_expression *) 0x7ffff1f5c0b0 s                                 1.
#       `--$nv3 (Query_block *) 0x7ffff1f5c820 #2, leaf tables: ($nx0)t3, ($nx1) t4
#          `--$nv4 (Query_expression *) 0x7ffff24f86e8 
#             `--$nv5 (Query_block *) 0x7ffff24f8e58 #3, leaf tables: ($ny0)t3
#             `--$nv5 (Query_block *) 0x7ffff24f9e58 #4d, leaf tables: ($nz0)t4    2.
# Here:
#  1. 's' means Item_singlerow_subselect, other position markers see get_subselect_marker()
#  2. '4d' 'd' means union_distinct
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
        marker = self.get_subselect_marker(val['item'])
        if marker:
            marker += ' '
        return marker + self.get_current_marker(val)

    def get_subselect_marker(self, subs):
        if not subs:
            return ''
        subs_class_name = subs.dynamic_type.target().name
        # 's' -> Item_singlerow_subselect
        # 'm' -> Item_maxmin_subselect
        # 'e' -> Item_exists_subselect
        # 'i' -> Item_in_subselect
        # 'a' -> Item_allany_subselect
        return subs_class_name[5]

    def show_Query_block(self, val):
        select_number = val['select_number']
        leaf_tables = val['leaf_tables']
        table_list = traverse_TABLE_LIST(leaf_tables, True)
        marker = 'd' if val['master']['union_distinct'] is not None and \
            val['master']['union_distinct'] == val else ''
        return "#%d%s, " % (select_number, marker) + table_list + self.get_current_marker(val)

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

def table_name_from_access_path(val):
    apfield = find_access_path_struct(val)
    aptyp_struct = val['u'][apfield.name]
    has_table = False;
    for field in aptyp_struct.type.fields():
        if field.name.lower() == 'table':
            has_table = True
            break
    if not has_table or not aptyp_struct['table']:
        return ''
    name = aptyp_struct['table']['s']['table_name']['str'].string()
    alias = aptyp_struct['table']['alias'].string()
    return f'<{name} {alias}>' if name.find('/') == -1 else f'<{alias}>'

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

    def get_children_from_array(self, array, array_item_field):
        size = int(array['m_size'])
        raw_array = array['m_array']
        aps = []
        for i in range(size):
            aps.append(raw_array[i][array_item_field])
        return aps

    def walk_AccessPath(self, val):
        apfield = find_access_path_struct(val)
        aptype = val['u'][apfield.name]
        child_aps = []
        for field in aptype.type.fields():
            if is_pointer_of(field, 'AccessPath') and aptype[field.name] != 0:
                child_aps.append(aptype[field.name])
            elif is_pointer_of(field, 'MaterializePathParameters'):
                child_aps += self.get_children_from_array(aptype[field]['query_blocks'], 'subquery_path')
            elif field.name == 'children':
                child_aps += self.get_children_from_array(aptype[field], 'path')

        return child_aps

    def show_AccessPath(self, val):
        apfield = find_access_path_struct(val)
        aptype = str(val['type']).split('::')[1]
        aptyp_struct = val['u'][apfield.name]
        struct_detail = str(aptyp_struct)
        return aptype + table_name_from_access_path(val) + ': ' + \
            self.autoncvar.set_var(aptyp_struct) + ' ' + struct_detail
AccessPathTraverser()

class OptPathTraverser(gdb.Command, TreeWalker):
    """explore access path tree"""
    def __init__ (self):
        super(self.__class__, self).__init__ ("mysql optpath", gdb.COMMAND_OBSCURE)

    def invoke(self, arg, from_tty):
        if not arg:
            print("usage: mysql optpath [path] [context]")
            return
        args = arg.split()
        path = gdb.parse_and_eval(args[0])
        self.context = gdb.parse_and_eval(args[1])
        self.walk(path)

    def walk_pq_opt_Path(self, val):
        subpaths = []
        subpath1 = val['m_subpath1']
        if subpath1:
            subpaths.append(subpath1)
        else:
            return subpaths
        subpath2 = val['m_subpath2']
        if subpath2:
            subpaths.append(subpath2)
        return subpaths

    def distinfo_to_str(self, distinfo):
        disttype = str(distinfo['type'])
        return disttype.split('::')[-1]

    def get_access_path(self, apindex):
        array = self.context['m_access_path_state']['m_array']
        return array[apindex]

    def show_pq_opt_Path(self, val):
        distinfo = val['m_distinfo']
        self.distinfo_to_str(distinfo)
        apindex = int(val['m_access_path_state'])
        access_path = self.get_access_path(apindex)['access_path']
        aptype = str(access_path['type']).split('::')[-1]
        aptables = table_name_from_access_path(access_path)
        return "{} ({}){}: {}".format(
            aptype, self.autoncvar.set_var(access_path), aptables,
            self.distinfo_to_str(distinfo))
OptPathTraverser()

class MDL_traverser(gdb.Command):
    """print metadata locks"""
    def __init__ (self):
        super (MDL_traverser, self).__init__("mysql mdl", gdb.COMMAND_OBSCURE)

    def _print_mdl_list(self, args):
        mdl = args.dereference()
        while str(mdl.address) != '0x0':
            mdl_dict = {}
            mdl_dict['mdl_type'] = str(mdl['m_type'])
            mdl_dict['mdl_key_mptr'] = str(mdl['m_lock']['key']['m_ptr'])
            mdl_dict['mdl_key_type'] = str(mdl['m_lock']['key']['m_ptr'][0].cast(gdb.lookup_type("MDL_key::enum_mdl_namespace")))
            print(str(mdl_dict))
            mdl = mdl['next_in_context'].dereference()

    def invoke(self, args, from_tty):
        if not args:
            print("usage: mysql mdl [THD]")
        thd_val = gdb.parse_and_eval(args).dereference()
        if str(thd_val.type).startswith("THD"):
            print("m_durations[0]")
            self._print_mdl_list(thd_val['mdl_context']['m_ticket_store']['m_durations'][0]['m_ticket_list']['m_first'])
            print("m_durations[1]")
            self._print_mdl_list(thd_val['mdl_context']['m_ticket_store']['m_durations'][1]['m_ticket_list']['m_first'])
            print("m_durations[2]")
        else:
            print("usage: mysql mdl [THD]")
MDL_traverser()

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

class Mem_root_arrayPrinter(object):
    """Print List a Mem_root_array"""

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

class Prealloced_arrayPrinter(object):
    """Print List a Prealloced_array"""

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
        inline_size = self.val['m_inline_size']
        return self._iterator(nodetype, self.val['m_buff'] if inline_size >= 0 else self.val['m_ext']['m_array_ptr'], self.size())

    def size(self):
        inline_size = self.val['m_inline_size']
        return inline_size if inline_size >= 0 else self.val['m_ext']['m_alloced_size']

    def to_string(self):
        return '%s' % self.typename if self.size() != 0 else 'empty %s' % self.typename

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

class Func_ptrPrinter(object):
    """Print actual Item type in Func_ptr"""
    def __init__(self, val):
        self.val = val

    def to_string(self):
        copy_from = self.val['m_func']
        copy_to = self.val['m_result_field']
        bits = self.val['m_func_bits']
        bits_str = ''
        cft = gdb.lookup_type('Copy_func_type')
        for bit in cft.fields():
            mask = 1 << bit.enumval
            if mask & bits:
                bits_str += bit.name + ' | '
        bits_str = bits_str[:-3]

        return '(%s) %s -> (%s) %s with %s' % (
            copy_from.dynamic_type, copy_from, copy_to.dynamic_type, copy_to, bits_str)

class mysql_mutex_t_Printer(object):
    """Print mysql_mutex_t, just print a native lock number"""
    def __init__(self, val):
        self.val = val

    def to_string(self):
        return "{lock = %d}" % self.val['m_mutex']['m_u']['m_native']['__data']['__lock']

import gdb.printing
def build_pretty_printer():
    pp = gdb.printing.RegexpCollectionPrettyPrinter(
        "mysqld")
    pp.add_printer('List', '^List<.*>$', ListPrinter)
    pp.add_printer('mem_root_deque', '^mem_root_deque<.*>$', mem_root_dequePrinter)
    pp.add_printer('AccessPath', '^AccessPath$', AccessPathPrinter)
    pp.add_printer('Mem_root_array', '^Mem_root_array<.*>$', Mem_root_arrayPrinter)
    pp.add_printer('Prealloced_array', '^Prealloced_array<.*>$', Prealloced_arrayPrinter)
    pp.add_printer('Bounds_checked_array', '^Bounds_checked_array<.*>$', Bounds_checked_arrayPrinter)
    pp.add_printer('Func_ptr', '^Func_ptr$', Func_ptrPrinter)
    pp.add_printer('mysql_mutex_t', '^mysql_mutex_t$', mysql_mutex_t_Printer)
    return pp

gdb.printing.register_pretty_printer(
    gdb.current_objfile(),
    build_pretty_printer(),
    True)
