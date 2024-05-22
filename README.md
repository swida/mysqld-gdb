A gdb module for mysql-server development
=========================================
Defines some gdb commands and pretty printers.

Commands
========
- thread overview -- display thread brief with backtrace aggregation
- thread search -- find a thread given a keyword of backtrace function name
- mysql digest -- decode a statement digest to readable string
- mysql item -- explore expression (Item) tree
- mysql queryblock -- explore query block tree
- mysql seltree -- explore SEL_TREE struct
- mysql tablelist -- traverse TABLE_LIST list
- mysql accesspath -- explore AccessPath tree

Pretty printers
===============

- List
- mem_root_deque
- AccessPath
- Mem_root_array
- Bounds_checked_array (e.g. base_ref_items)
- Func_ptr

Requirement
===========

My development environment:

- gdb version 12.1
- MySQL server version 8.0.26

Installation
============
1. Copy autocvar.py to the `PYTHON_PATH` directory
  e.g.`/usr/share/gdb/python`
2. copy `mysqld-gdb.py` to the directory same with mysqld executable file.
3. Add mysqld executable file path to gdb safe path dy adding this to
  `~/.gdbinit` (replace `<mysqld executable file path>` with your real
  path):

```
set auto-load safe-path <mysqld executable file path>
```

Usage Examples
==============

Pretty Printers
---------------
```
(gdb) p *join->fields
$93 = mem_root_deque<Item*> = {$ae0 = (Item_field *) 0x7ffee8080cf0, $ae1 = (Item_sum_count *) 0x7ffee8081680}
(gdb) p $ae0->hidden
$94 = false
```
GDB MySQL Commands
------------------
```
(gdb) my acc source_join->m_root_access_path
$ai0 (AccessPath *) 0x7ffee808ab38 MATERIALIZE $ai1 {table_path = 0x7ffee808aa28, param = 0x7ffee808aae0}
|--$ai2 (AccessPath *) 0x7ffee808aa28 TABLE_SCAN $ai3 { table = <temporary> }
`--$ai4 (AccessPath *) 0x7ffee807b508 TEMPTABLE_AGGREGATE $ai5 {subquery_path = 0x7ffee807b3f8, temp_table_param = 0x7ffee8066b40, table = 0x7ffee804eaa8, table_path = 0x7ffee807b480, ref_slice = 1}
   |--$ai6 (AccessPath *) 0x7ffee807b3f8 FILTER $ai7 {child = 0x7ffee807b2c0, condition = 0x7ffee8064ff0, materialize_subqueries = false}
   |  `--$ai8 (AccessPath *) 0x7ffee807b2c0 TABLE_SCAN $ai9 { table = t1 }
   `--$ai10 (AccessPath *) 0x7ffee807b480 TABLE_SCAN $ai11 { table = <temporary> }
(gdb) my item $ai7.condition
$aj0 (Item_func_gt *) 0x7ffee8064ff0
|--$aj1 (Item_field *) 0x7ffee8064ea0 field = test.t1.a
`--$aj2 (Item_int *) 0x7ffee800deb8 value = 5
```
