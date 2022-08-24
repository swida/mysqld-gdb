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
- mem_root_array

Requirement
===========
- gdb version 12.1
- MySQL server version 8.0.26
