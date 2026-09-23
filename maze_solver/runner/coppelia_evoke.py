"""
Launch CoppeliaSim in its own console and guarantee it dies with us.

`Popen.terminate()` alone is not enough on Windows:

  * it only kills the process we spawned -- CoppeliaSim's own children
    (and the conhost that owns the CREATE_NEW_CONSOLE window) survive;
  * it never runs at all when this process is killed hard (taskkill,
    Ctrl+Break, a crash, the IDE's stop button), because no `finally`
    block executes -- and CoppeliaSim then stays up forever, holding the
    ZMQ remote-api port so the *next* run cannot connect.

So the child is put in a Job Object created with
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE. The job's only handle lives in this
process; the kernel closes it whenever this process ends, however it
ends, and immediately terminates every process still in the job.
"""
import os
import psutil
import ctypes
import subprocess
from ctypes import wintypes

from src.definitions import PATHS

_IS_WINDOWS = os.name == "nt"

# Handle of the job the children live in. Module-level on purpose: if this
# is garbage-collected the job closes and CoppeliaSim is killed with it.
_JOB = None

_JobObjectExtendedLimitInformation = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
        ("PerJobUserTimeLimit",     wintypes.LARGE_INTEGER),
        ("LimitFlags",              wintypes.DWORD),
        ("MinimumWorkingSetSize",   ctypes.c_size_t),
        ("MaximumWorkingSetSize",   ctypes.c_size_t),
        ("ActiveProcessLimit",      wintypes.DWORD),
        ("Affinity",                ctypes.c_size_t),
        ("PriorityClass",           wintypes.DWORD),
        ("SchedulingClass",         wintypes.DWORD),
    ]


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount",  "WriteTransferCount",  "OtherTransferCount")]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo",                _IO_COUNTERS),
        ("ProcessMemoryLimit",    ctypes.c_size_t),
        ("JobMemoryLimit",        ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed",     ctypes.c_size_t),
    ]


def _kill_on_close_job():
    """Create (once) a job whose members are killed when this process dies."""
    global _JOB
    if _JOB is not None:
        return _JOB

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, wintypes.INT,
                                            wintypes.LPVOID, wintypes.DWORD]

    # NULL security attributes -> non-inheritable handle, so a child cannot
    # keep the job (and therefore itself) alive after we are gone.
    job = k32.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())

    info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not k32.SetInformationJobObject(job, _JobObjectExtendedLimitInformation,
                                       ctypes.byref(info), ctypes.sizeof(info)):
        err = ctypes.get_last_error()
        k32.CloseHandle(job)
        raise ctypes.WinError(err)

    _JOB = job
    return job


def _adopt(process: subprocess.Popen) -> bool:
    """Put `process` (and every process it spawns) in the kill-on-close job."""
    process_pid = (int(process._handle) 
                   if isinstance(process,subprocess.Popen) 
                   else process)
    
    if not _IS_WINDOWS:
        return False
    
    try:
        job = _kill_on_close_job()
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        if not k32.AssignProcessToJobObject(job, process_pid):
            raise ctypes.WinError(ctypes.get_last_error())
        return True
    
    except OSError as exc:
        # Not fatal: we simply fall back to the best-effort terminate below.
        print(f"[WARN] nao foi possivel prender o CoppeliaSim ao job: {exc}")
        return False

def  _is_coppelia_runing():
    for proc in psutil.process_iter(["pid","name"]):
        if "coppeliaSim" in proc.info["name"]:
            return int(proc.info["pid"])
    return None

# Start CoppeliaSim and Leaves, raise EVOKE_ERROR
def evoke_coppelia(should_adopt:bool) -> subprocess.Popen:
    creationflags = 0
    
    process = _is_coppelia_runing()
    
    if not process:
        _flags = subprocess.CREATE_BREAKAWAY_FROM_JOB if should_adopt else 0x00
        if _IS_WINDOWS:
            # CREATE_BREAKAWAY_FROM_JOB lets us re-parent the child into our own
            # job even when this process was itself started inside one that does
            # not allow silent nesting.
            creationflags = (subprocess.CREATE_NEW_CONSOLE | _flags)

        try:
            process = subprocess.Popen([PATHS.COPPELIA_BINARY],
                                    creationflags=creationflags)
        except OSError:
            # The current job forbids breakaway -- retry without it.
            process = subprocess.Popen([PATHS.COPPELIA_BINARY],
                                    creationflags=subprocess.CREATE_NEW_CONSOLE
                                    if _IS_WINDOWS else 0)

    if should_adopt:
        _adopt(process)
    
    return process


def terminate_coppelia(process: subprocess.Popen, timeout: float = 10.0) -> None:
    """Close CoppeliaSim and its console on the normal shutdown path.

    The job object already covers the abnormal paths; this just gives the
    simulator a chance to exit cleanly while we are still alive.
    """
    _SIGTERM = 15
    if isinstance(process,int):
        os.kill(process,_SIGTERM)
        if _IS_WINDOWS:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(process)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            check=False)
        return

    if process is None or process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass

    if _IS_WINDOWS:
        # Whole tree, forced -- CoppeliaSim leaves children behind.
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=False)
    else:
        process.kill()

    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        print("[WARN] CoppeliaSim nao encerrou; o job o matara na saida")


def load_scene_in_coppelia(sim):
    sim.loadScene(PATHS.KART_SHADOW)
