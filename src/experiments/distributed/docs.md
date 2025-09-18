got it — here’s the no-nonsense docs your teammates need to build **their own** subscriber (worker) and publisher (router) scripts on top of your DIT + NATS setup.

# How to build a Subscriber script (worker)

### Purpose

Run one expert (`model_id`) that listens on `models.<model_id>`, executes a callable model, and **always replies** with a protobuf `Response`.

### You must implement

1. **Inputs / CLI args**

   * `--model-id` (required): expert id; defines the subject `models.<model_id>`.
   * `--nats-url` (default `nats://host:4222`): same for all processes.
   * Optional: `--queue-group` (default `ditq.<model_id>`), `--max-inflight` (default 64).
   * If using HF: `--task`, `--model-name`.

2. **Model load**

   * Use your `DitExpert`:

     * Either `DitExpert(model=<callable>)`, or
     * `expert.load_model(task=..., model_name=...)` for HF.

3. **NATS subscribe**

   * Connect once.
   * `subscribe("models.<model_id>", queue="ditq.<model_id>", cb=handler)`.

4. **Handler contract**

   * Parse `Request` from `msg.data`.
   * Call `expert.run_model(req.payload)` (catch exceptions).
   * Build `Response` with:

     * `request_id` (echo from req),
     * `model_id`,
     * `payload` (stringify output),
     * `response_status` (SUCCESS/ERROR),
     * `latency_ms`,
     * `error_message` on errors.
   * **Always** `await msg.respond(resp.SerializeToString())` (even on parse/model errors).

5. **Concurrency**

   * Guard with `asyncio.Semaphore(max_inflight)`.

6. **Logging**

   * Log when subscribed; log each request and reply (status + latency).


### Minimal flow

```
parse -> run_model -> build Response -> respond -> log
```

---

# How to build a Publisher script (router/orchestrator)

### Purpose

Build a `DIT` whose experts are **remote callables** (NATS request/reply), route queries with your router (e.g., `SimpleRouter`), and collect replies.

### You must implement

1. **Inputs / CLI args**

   * `--experts` (list): expert IDs the router can choose.
   * `--nats-url` (default `nats://host:4222`).
   * `--queries` or `--query`/`--queries-file`.
   * `--timeout-ms` (e.g., 800) and `--retries` (e.g., 0–2).

2. **Publisher transport**

   * Use a **single background event loop** for NATS:

     * One persistent NATS connection inside a `Publisher`.
     * Expose `ask_sync(model_id, payload)` that marshals to/from protobuf and honors timeout/retries.
   * (This avoids the “first request works, then timeouts” loop mismatch.)

3. **Remote expert binding**

   * For each expert id, set `DitExpert.model = make_remote_callable(publisher, model_id)`, where the callable simply:

     ```
     resp = publisher.ask_sync(model_id, query)
     if resp.status != SUCCESS: raise RuntimeError(resp.error_message)
     return resp.payload
     ```

4. **Router + DIT**

   * `router = SimpleRouter(experts=[...])`
   * `dit = DIT(experts=table, router=router)`

5. **Send queries**

   * For each query:

     * Optional: time `router.route(q)` (routing cost).
     * Call `dit.exec(q)` (unchanged API).
     * Write result or error to CSV / print logs.

6. **Logging**

   * Log `[PUB] -> subject=models.<id> ...` before send.
   * Log `[PUB] <- status=... latency=...` after reply.

### Do / Don’t

*  Do keep **timeouts short** and retries small during dev.
*  Do restrict `--experts` to IDs that actually have workers running.
* Don’t create a new event loop per call (`asyncio.run` per request is bad).
*  Don’t reuse a NATS connection across different loops.

### Minimal flow

```
build Publisher -> bind remote experts -> DIT(router) -> for q: dit.exec(q) -> log/save
```

---

# Subject & Queue naming (must match)

* **Subject**: `models.<model_id>` (case-sensitive).
* **Queue group**: `ditq.<model_id>` (all replicas share the same group to load-balance).

---

# Protobuf I/O (both scripts)

* Serialize requests: `req.SerializeToString()`
* Parse on subscriber: `req.ParseFromString(msg.data)`
* Serialize responses: `resp.SerializeToString()`
* Parse on publisher after request: `resp.ParseFromString(msg.data)`

Response fields you should set:

* `request_id` (copy from request)
* `model_id` (subscriber’s id)
* `payload` (string)
* `response_status` (SUCCESS / ERROR)
* `latency_ms` (int)
* `error_message` (string; empty if none)

---

# CLI patterns to copy

**Subscriber**

```bash
python3 worker_hf.py \
  --model-id Payments \
  --nats-url nats://127.0.0.1:4222 \
  --task text-classification \
  --model-name distilbert-base-uncased-finetuned-sst-2-english
```

**Publisher**

```bash
python3 pub_simple.py \
  --experts Payments Search Support \
  --nats-url nats://127.0.0.1:4222 \
  --queries 50 \
  --timeout-ms 800 \
  --retries 0
```

---

# Testing checklist (copy/paste)

1. Start NATS: `nats-server -m 8222`
2. Start workers for all `--experts` you’ll publish to.
3. Open `http://localhost:8222/subsz?subs=1` and confirm you see:

   * `subject: "models.<id>"`, `queue_name: "ditq.<id>"`, `num_subscriptions >= 1`
4. Run the publisher with short timeouts, 0 retries.
5. Expect to see logs on both sides (`got payload` on sub, `status=SUCCESS` on pub).
6. CSV written under your data dir with responses, not timeouts.

---

# Common failures → fixes

* **`no responders (no subscriber ...)`**
  You routed to an expert without a running worker (or different broker). Start worker or adjust `--experts`. Verify on `/subsz?subs=1`.

* **Timeouts after the first request**
  You used `asyncio.run()` per call. Fix by using a **single background loop** in Publisher and `ask_sync`.

* **Subscriber prints “got payload” but pub times out**
  Your handler crashed before responding. Wrap the entire handler with try/except; **always** `await msg.respond(...)` even on errors; log any “respond failed”.

* **`MessageFactory.GetPrototype`**
  Protobuf v5 runtime with v4 stubs. Pin runtime: `protobuf==4.25.3` and regenerate stubs with the same interpreter.

---

# What to include when you create new scripts

* **CLI args:** `--model-id` (sub), `--experts` (pub), shared `--nats-url`, timeouts/retries.
* **Imports:** `protoc` stubs, `DitExpert`, `DIT`, your router(s), `Publisher`/`Subscriber`.
* **Lifecycle:** connect once, reuse connections/loops, graceful `drain()` on shutdown.
* **Observability:** minimal prints for send/recv; optional CSV output in pub.
* **Resilience:** catch all exceptions in handlers; never drop a request without a reply.

That’s it — mirror this checklist and you’ll have reliable, composable sub/pub scripts every time.