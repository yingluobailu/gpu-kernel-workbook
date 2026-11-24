"""Time a prepared launch with start and stop event nodes in a CUDA graph."""

from contextlib import ExitStack, contextmanager
import ctypes as c

from workbook.cuda import Handle, Int, Ref, UInt, bind


@contextmanager
def timed_graph(device, launch):
    lib = device.lib
    check = device.check
    stream_create = bind(lib, "cuStreamCreate", Ref(Handle), UInt)
    stream_destroy = bind(lib, "cuStreamDestroy_v2", Handle)
    stream_sync = bind(lib, "cuStreamSynchronize", Handle)
    event_create = bind(lib, "cuEventCreate", Ref(Handle), UInt)
    event_destroy = bind(lib, "cuEventDestroy_v2", Handle)
    event_record = bind(lib, "cuEventRecordWithFlags", Handle, Handle, UInt)
    elapsed = bind(lib, "cuEventElapsedTime", Ref(c.c_float), Handle, Handle)
    begin = bind(lib, "cuStreamBeginCapture_v2", Handle, Int)
    end = bind(lib, "cuStreamEndCapture", Handle, Ref(Handle))
    destroy = bind(lib, "cuGraphDestroy", Handle)
    instantiate = bind(lib, "cuGraphInstantiateWithFlags", Ref(Handle), Handle, c.c_ulonglong)
    exec_destroy = bind(lib, "cuGraphExecDestroy", Handle)
    replay = bind(lib, "cuGraphLaunch", Handle, Handle)

    with ExitStack() as resources:
        stream = Handle()
        check(stream_create(c.byref(stream), 1), "Create nonblocking benchmark stream")
        resources.callback(stream_destroy, stream)
        start, stop = Handle(), Handle()
        for event in (start, stop):
            check(event_create(c.byref(event), 0), "Create timing event")
            resources.callback(event_destroy, event)

        graph, executable = Handle(), Handle()
        check(begin(stream, 1), "Begin thread-local capture")
        try:
            # EXTERNAL retains both timestamps as graph nodes during replay.
            check(event_record(start, stream, 1), "Capture start timestamp")
            device.launch(*launch, stream=stream)
            check(event_record(stop, stream, 1), "Capture stop timestamp")
        finally:
            status = end(stream, c.byref(graph))
            if graph:
                resources.callback(destroy, graph)
        check(status, "Finish benchmark capture")
        check(instantiate(c.byref(executable), graph, 0), "Instantiate benchmark graph")
        resources.callback(exec_destroy, executable)

        def sample_us():
            check(replay(executable, stream), "Replay benchmark graph")
            check(stream_sync(stream), "Complete benchmark graph")
            ms = c.c_float()
            check(elapsed(c.byref(ms), start, stop), "Read GPU time")
            return ms.value * 1000

        yield sample_us
