[New LWP 2681]
[New LWP 2680]
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
__memset_avx2_unaligned_erms () at ../sysdeps/x86_64/multiarch/memset-vec-unaligned-erms.S:328

warning: 328	../sysdeps/x86_64/multiarch/memset-vec-unaligned-erms.S: No such file or directory

Thread 3 (Thread 0x7ff0027ff6c0 (LWP 2680) "python3"):
#0  0x00007ff009e98e51 in __futex_abstimed_wait_common64 (private=0, cancel=true, abstime=0x0, op=393, expected=0, futex_word=0x7ff008bbd660 <thread_status+96>) at ./nptl/futex-internal.c:57
#1  __futex_abstimed_wait_common (cancel=true, private=0, abstime=0x0, clockid=0, expected=0, futex_word=0x7ff008bbd660 <thread_status+96>) at ./nptl/futex-internal.c:87
#2  __GI___futex_abstimed_wait_cancelable64 (futex_word=futex_word@entry=0x7ff008bbd660 <thread_status+96>, expected=expected@entry=0, clockid=clockid@entry=0, abstime=abstime@entry=0x0, private=private@entry=0) at ./nptl/futex-internal.c:139
#3  0x00007ff009e9b8cd in __pthread_cond_wait_common (abstime=0x0, clockid=0, mutex=0x7ff008bbd610 <thread_status+16>, cond=0x7ff008bbd638 <thread_status+56>) at ./nptl/pthread_cond_wait.c:503
#4  ___pthread_cond_wait (cond=0x7ff008bbd638 <thread_status+56>, mutex=0x7ff008bbd610 <thread_status+16>) at ./nptl/pthread_cond_wait.c:627
#5  0x00007ff007a19ad3 in blas_thread_server () from /opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/site-packages/numpy/_core/../../numpy.libs/libscipy_openblas64_-f48b354e.so
#6  0x00007ff009e9cb84 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:447
#7  0x00007ff009f29ecc in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:78

Thread 2 (Thread 0x7ff0017fe6c0 (LWP 2681) "memopro-pager"):
#0  0x00007ff009f1b79d in __GI___poll (fds=0x7ff0017fd488, nfds=2, timeout=-1) at ../sysdeps/unix/sysv/linux/poll.c:29
#1  0x00007ff00a996afd in std::sys::backtrace::__rust_begin_short_backtrace::<<memopro::rt::pager::imp::Pager>::new::{closure#3}, ()> () from /home/runner/work/memopro/memopro/target/release/libmemopro_preload.so
#2  0x00007ff00a999204 in <std::thread::lifecycle::spawn_unchecked<<memopro::rt::pager::imp::Pager>::new::{closure#3}, ()>::{closure#1} as core::ops::function::FnOnce<()>>::call_once::{shim:vtable#0} () from /home/runner/work/memopro/memopro/target/release/libmemopro_preload.so
#3  0x00007ff00a9d0300 in <std::sys::thread::unix::Thread>::new::thread_start () from /home/runner/work/memopro/memopro/target/release/libmemopro_preload.so
#4  0x00007ff009e9cb84 in start_thread (arg=<optimized out>) at ./nptl/pthread_create.c:447
#5  0x00007ff009f29ecc in clone3 () at ../sysdeps/unix/sysv/linux/x86_64/clone3.S:78

Thread 1 (Thread 0x7ff00a925800 (LWP 2678) "python3"):
#0  __memset_avx2_unaligned_erms () at ../sysdeps/x86_64/multiarch/memset-vec-unaligned-erms.S:328
#1  0x00007ff008e756c5 in _aligned_strided_to_contig_size1_srcstride0 () from /opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/site-packages/numpy/_core/_multiarray_umath.cpython-312-x86_64-linux-gnu.so
#2  0x00007ff008f7e73f in raw_array_assign_scalar () from /opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/site-packages/numpy/_core/_multiarray_umath.cpython-312-x86_64-linux-gnu.so
#3  0x00007ff008f7f13d in PyArray_AssignRawScalar () from /opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/site-packages/numpy/_core/_multiarray_umath.cpython-312-x86_64-linux-gnu.so
#4  0x00007ff008ff8ade in array_copyto () from /opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/site-packages/numpy/_core/_multiarray_umath.cpython-312-x86_64-linux-gnu.so
#5  0x00007ff00a4344e3 in cfunction_vectorcall_FASTCALL_KEYWORDS (func=0x7ff009992340, args=0x7ff00a8a7120, nargsf=<optimized out>, kwnames=0x7ff008dfd8a0) at ./Include/cpython/methodobject.h:50
#6  0x00007ff00a4145d3 in _PyObject_VectorcallTstate (kwnames=0x7ff008dfd8a0, nargsf=9223372036854775810, args=0x7ff00a8a7120, callable=0x7ff009992340, tstate=0x7ff00a88c6f0 <_PyRuntime+458992>) at ./Include/internal/pycore_call.h:92
#7  PyObject_Vectorcall (callable=0x7ff009992340, args=0x7ff00a8a7120, nargsf=9223372036854775810, kwnames=0x7ff008dfd8a0) at Objects/call.c:325
#8  0x00007ff008f811e8 in dispatcher_vectorcall () from /opt/hostedtoolcache/Python/3.12.14/x64/lib/python3.12/site-packages/numpy/_core/_multiarray_umath.cpython-312-x86_64-linux-gnu.so
#9  0x00007ff00a4145d3 in _PyObject_VectorcallTstate (kwnames=0x7ff008dfd8a0, nargsf=9223372036854775810, args=0x7ff00a8a7120, callable=0x7ff008ddbc30, tstate=0x7ff00a88c6f0 <_PyRuntime+458992>) at ./Include/internal/pycore_call.h:92
#10 PyObject_Vectorcall (callable=0x7ff008ddbc30, args=0x7ff00a8a7120, nargsf=9223372036854775810, kwnames=0x7ff008dfd8a0) at Objects/call.c:325
#11 0x00007ff00a38d26e in _PyEval_EvalFrameDefault (tstate=<optimized out>, frame=0x7ff00a8a7098, throwflag=<optimized out>) at Python/bytecodes.c:2715
#12 0x00007ff00a4e8e06 in PyEval_EvalCode (co=co@entry=0x7ff009920170, globals=globals@entry=0x7ff00a10e9c0, locals=locals@entry=0x7ff00a10e9c0) at Python/ceval.c:580
#13 0x00007ff00a509e47 in run_eval_code_obj (tstate=tstate@entry=0x7ff00a88c6f0 <_PyRuntime+458992>, co=co@entry=0x7ff009920170, globals=globals@entry=0x7ff00a10e9c0, locals=locals@entry=0x7ff00a10e9c0) at Python/pythonrun.c:1757
#14 0x00007ff00a50997b in run_mod (mod=mod@entry=0x561163ded8d8, filename=filename@entry=0x7ff00a822460 <_PyRuntime+24160>, globals=globals@entry=0x7ff00a10e9c0, locals=locals@entry=0x7ff00a10e9c0, flags=flags@entry=0x7ffc239a92d0, arena=arena@entry=0x7ff00a033e50) at Python/pythonrun.c:1778
#15 0x00007ff00a5094cd in PyRun_StringFlags (str=<optimized out>, start=<optimized out>, globals=0x7ff00a10e9c0, locals=0x7ff00a10e9c0, flags=0x7ffc239a92d0) at Python/pythonrun.c:1649
#16 0x00007ff00a50941c in PyRun_SimpleStringFlags (command=0x7ff00a0607c0 "import numpy as np; a = np.ones(64 << 20, np.uint8); print(int(a.sum()))\n", flags=flags@entry=0x7ffc239a92d0) at Python/pythonrun.c:506
#17 0x00007ff00a512965 in pymain_run_command (command=<optimized out>) at Modules/main.c:255
#18 pymain_run_python (exitcode=0x7ffc239a92a4) at Modules/main.c:625
#19 Py_RunMain () at Modules/main.c:714
#20 0x00007ff00a51230b in Py_BytesMain (argc=<optimized out>, argv=<optimized out>) at Modules/main.c:768
#21 0x00007ff009e2a1ca in __libc_start_call_main (main=main@entry=0x56115f255060 <main>, argc=argc@entry=3, argv=argv@entry=0x7ffc239a9538) at ../sysdeps/nptl/libc_start_call_main.h:58
#22 0x00007ff009e2a28b in __libc_start_main_impl (main=0x56115f255060 <main>, argc=3, argv=0x7ffc239a9538, init=<optimized out>, fini=<optimized out>, rtld_fini=<optimized out>, stack_end=0x7ffc239a9528) at ../csu/libc-start.c:360
#23 0x000056115f255095 in _start ()
[Inferior 1 (process 2678) detached]
