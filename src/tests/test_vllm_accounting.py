"""Offline regressions. Synthetic ledgers never certify saved runs as K8s runs."""
from __future__ import annotations
import asyncio
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.backends.vllm import accounting as a, config, inputs, ledger, reference_accounting as ref
from src.backends.vllm.models import tar_directory

FIXTURES = Path(__file__).parent / 'fixtures' / 'vllm'


def cases():
    return json.loads(gzip.decompress((FIXTURES/'accounting-saved.json.gz').read_bytes()))


def case_run(case, model):
    raw = case['run']
    return ref.Run(**{**raw, 'model': model, 'proxy': Path(raw['proxy']),
                      'prefill': ref.Side(**raw['prefill']), 'decode': ref.Side(**raw['decode']),
                      'server': ref.Server(**raw['server']), 'workload': ref.Workload(**raw['workload'])})


def logs_of(case):
    return {tuple(item['key']): item['text'] for item in case['logs']}


def canonical(value):
    return json.loads(json.dumps(value))


def seal(root):
    path = root/'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['artifact_hashes'] = {str(p.relative_to(root)): inputs.sha256(p) for p in root.rglob('*')
                                   if p.is_file() and p.name not in ('manifest.json','accounting.json')}
    path.write_text(json.dumps(manifest))


def synthetic_capture(root, case):
    """Attach fabricated *test-only* lifecycle receipts to real saved geometry.

    This tests the raw regeneration contract. The reference saved run has no
    such client/proxy ledger or K8s epoch and is never reported to have one.
    """
    def save(name, obj):
        (root/name).write_text(json.dumps(obj))
    model = root/'inputs'/'model';model.mkdir(parents=True)
    (model/'config.json').write_text(json.dumps(case['model_config']))
    payload = tar_directory(model)
    (root/'inputs'/'model.tar').write_bytes(payload)
    model_hash = inputs.sha256(model/'config.json')
    receipt = {'model':{'sha256':{'config.json':model_hash},'archive_sha256':hashlib.sha256(payload).hexdigest(),'artifact':'model.tar'}}
    save('inputs/receipt.json',receipt)

    requested_image='example/sim@sha256:'+'a'*64
    pod_identities={}
    for role,node,pod in (('prefill','prefill-node','p'),('decode','decode-node','d')):
        identity={'pod':pod,'namespace':'n','uid':f'synthetic-{role}-pod','node':node,
                  'container_id':f'synthetic-{role}-container','image_requested':requested_image,
                  'image_id':requested_image,'pod_ip':'10.42.0.2' if role=='prefill' else '10.42.1.2'}
        pod_identities['target' if role=='prefill' else 'initiator']=identity
        for phase in ('before','after'):save(f'{role}-pod-{phase}.json',identity)
    save('deployment.json',{'status':'ready','image_requested':requested_image,'pods':pod_identities,'inputs':receipt})

    o = {**config.DEFAULTS, **case['run']['server'], 'connector':'nixl','nixl_backend':'libfabric',
         'kv_check':True,'timing':'zero','timing_args':{},'num_prompts':case['bench']['completed'],
         **{f'{role}_{key}':case['run'][role].get(key) or 1 for role in ('prefill','decode')
            for key in ('data_parallel_size','tensor_parallel_size')}}
    bench=deepcopy(case['bench'])
    # Real bench output contains exact client-clock starts and lengths.
    ids=sorted({a.request_uuid(req) for r in a.records(logs_of(case)[('decode',0,0,0)]) for req in (r.get('kv_events') or {}).get('allocated',{})})
    client, proxy = [], []
    for i, proxy_id in enumerate(ids):
        cid=f'synthetic-client-{i}'
        client.extend([{'event':'started','client_id':cid},
                       {'event':'finished','client_id':cid,'start_time':bench['start_times'][i],
                        'latency':bench['latencies'][i],'success':True,'error':bench['errors'][i],
                        'prompt_length':bench['input_lens'][i],'output_tokens':bench['output_lens'][i]}])
        proxy.extend([{'event':'mapped','client_id':cid,'proxy_id':proxy_id},
                      {'event':'finished','client_id':cid,'status':200}])
    save('client.json',{'exit_code':0,'bench':bench})
    (root/'client-ledger.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in client))
    (root/'proxy-ledger.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in proxy))
    physical=case['reference_verdict']['transfers']['bytes']
    metrics='\n'.join(f'{name}{{engine="0"}} 0' for name in ('vllm:kv_cache_usage_perc','vllm:num_requests_running','vllm:num_requests_waiting'))+'\n'
    metrics += ''.join(f'{name}{{engine="0"}} 0\n' for name in a.NIXL_COUNTERS)
    for role in ('prefill','decode'):
        text=logs_of(case)[(role,0,0,0)]
        rows=a.records(text)
        layout=next(r for r in rows if 'kv_layout' in r)
        startup=''.join(json.dumps(r)+'\n' for r in (layout,{'timing':{'provider':'zero','args':{}}},{'threads':{'1':'synthetic'}}))
        segment=''.join(json.dumps(r)+'\n' for r in rows if 'kv_layout' not in r)
        times=[r[k] for r in rows for k in ('t','t_start','t_deadline','t_scheduled') if k in r]
        before={'started':min(times)-2,'finished':min(times)-1,'clock':'node_monotonic','boot_id':f'{role}-test-boot',
                'server_processes':{'10':{'start_ticks':12}},
                'versions':{name:{'ok':True,'value':'synthetic-test-version'} for name in ('vllm','torch','nixl','libfabric')},
                'native_binaries_sha256':{'synthetic-library.so':{'ok':True,'value':'0'*64}},'metrics':{'ok':True,'value':metrics},
                'network':{'ok':True,'value':{'net1':[0,0,0,0]}},
                'ranks':{'dp0.tp0':{'steps':{'ok':True,'value':startup},'tent':{'applicable':False}}}}
        after=deepcopy(before);after.update(started=max(times)+1,finished=max(times)+2)
        after['ranks']['dp0.tp0']['steps']['value']=startup+segment
        if role == 'decode':
            descriptors=sum(entry[3] for row in rows for entry in ((row.get('kv_transfers') or {}).get('load') or {}).get('each',[]))
            for name,value in zip(a.NIXL_COUNTERS[:3],(physical,case['reference_verdict']['transfers']['count'],descriptors)):
                after['metrics']['value']=after['metrics']['value'].replace(f'{name}{{engine="0"}} 0\n',f'{name}{{engine="0"}} {value}\n')
        after['network']['value']['net1']=[physical+10000,100,physical+10000,100]
        save(f'{role}-before.json',before);save(f'{role}-after.json',after)
        save(f'{role}-input-hashes.json',{'/model/config.json':model_hash})
    save('manifest.json',{'options':o,'image_requested':requested_image,'expected_nodes':{'prefill':'prefill-node','decode':'decode-node'},'collection_errors':[],'expected_instances':{'prefill':1,'decode':1},
                          'expected_ranks':[{'role':role,'instance':0,'dp':0,'tp':0} for role in ('prefill','decode')],'interfaces':{'prefill':'net1','decode':'net1'},
                          'artifact_hashes':{},'profilers_stopped':True,'samplers_stopped':True,'client_cleanup':{'stopped':True}})
    seal(root)


class SavedAccountingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases=cases()

    def test_exact_reference_oracle_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            model=Path(temp)
            for case in self.cases[:8] + self.cases[9:11]:
                with self.subTest(case=case['name']):
                    (model/'config.json').write_text(json.dumps(case['model_config']))
                    run=case_run(case,model)
                    result=ref.check(case['bench'],[r['text'] for r in case['logs'] if r['key'][0]=='prefill'],
                                     [r['text'] for r in case['logs'] if r['key'][0]=='decode'],case['routers'],run,case['tent'],case['ports'])
                    self.assertEqual(canonical(result),case['reference_verdict'])

    def test_incomplete_saved_transfer_records_are_rejected(self):
        case=self.cases[11]
        with tempfile.TemporaryDirectory() as tmp:
            model=Path(tmp);(model/'config.json').write_text(json.dumps(case['model_config']))
            with self.assertRaisesRegex(ValueError,'seven fields'):
                a.core(case['bench'],logs_of(case),case_run(case,model),case['tent'])

    def test_incomplete_saved_read_plans_are_rejected(self):
        case=self.cases[8]
        with tempfile.TemporaryDirectory() as tmp:
            model=Path(tmp);(model/'config.json').write_text(json.dumps(case['model_config']))
            result=a.core(case['bench'],logs_of(case),case_run(case,model),case['tent'])
        self.assertTrue(any('selected bytes and lengths' in p for p in result['problems']))

    def test_grouped_layout_checks_and_explicit_dp_identity_are_required(self):
        case=self.cases[2]
        with tempfile.TemporaryDirectory() as tmp:
            model=Path(tmp);(model/'config.json').write_text(json.dumps(case['model_config']))
            run=case_run(case,model)
            for kind,field,error in (('kv_layout','groups','grouped KV layout'),
                                     ('kv_check','groups','grouped KV check'),
                                     ('kv_layout','dp_rank','wrong rank identity')):
                with self.subTest(kind=kind,field=field):
                    logs=logs_of(case);key=('decode',0,0,0);rows=a.records(logs[key])
                    next(row[kind] for row in rows if kind in row).pop(field)
                    logs[key]='\n'.join(json.dumps(row) for row in rows)+'\n'
                    with self.assertRaisesRegex(ValueError,error):
                        a.core(case['bench'],logs,run)

    def test_modern_rank_aggregation_and_byte_partition(self):
        with tempfile.TemporaryDirectory() as temp:
            model=Path(temp)
            for case in self.cases[:8]:
                with self.subTest(case=case['name']):
                    (model/'config.json').write_text(json.dumps(case['model_config']))
                    result=a.core(case['bench'],logs_of(case),case_run(case,model),case['tent'])
                    self.assertEqual(result['problems'],[])
                    self.assertEqual(result['requests'],40)
                    self.assertEqual(result['physical_transfers'],case['reference_verdict']['transfers'])
                    b=result['bytes']
                    self.assertTrue(b['available'])
                    self.assertEqual(b['physical_payload_bytes'],sum(b[k] for k in ('useful_kv_bytes','replicated_useful_bytes','transferred_padding_bytes','transfer_only_bytes')))
                    if 'dsv3' in case['name']:
                        self.assertEqual(b['replicated_useful_bytes'],3*b['useful_kv_bytes'])
                    if 'nemotron' in case['name']:
                        self.assertGreater(b['transfer_only_bytes'],0)
                        self.assertGreater(b['untransferred_page_padding_bytes'],0)
                    if 'qwen3.5' in case['name']:
                        self.assertGreater(b['transferred_padding_bytes'],0)
                        self.assertGreater(b['untransferred_page_padding_bytes'],0)

    def test_known_tent_negative_stays_invalid_despite_40_successes(self):
        case=next(c for c in self.cases if c['name'].endswith('153617'))
        with tempfile.TemporaryDirectory() as tmp:
            model=Path(tmp);(model/'config.json').write_text(json.dumps(case['model_config']))
            result=a.core(case['bench'],logs_of(case),case_run(case,model),case['tent'],case['routers'],case['ports'])
        self.assertEqual(result['requests'],40)
        self.assertEqual(result['physical_transfers']['failed'],0)
        self.assertTrue(any('2592' in problem for problem in result['problems']))

    def test_missing_duplicate_wrong_rank_and_missing_check(self):
        case=self.cases[3]
        with tempfile.TemporaryDirectory() as tmp:
            model=Path(tmp);(model/'config.json').write_text(json.dumps(case['model_config']))
            run=case_run(case,model);logs=logs_of(case);key=('decode',0,0,1)
            bad=dict(logs);bad.pop(key)
            with self.assertRaisesRegex(ValueError,'inventory'):a.core(case['bench'],bad,run)
            bad=dict(logs);bad[key]=bad[('decode',0,0,0)]
            with self.assertRaisesRegex(ValueError,'wrong rank'):a.core(case['bench'],bad,run)
            bad=dict(logs);bad[key]+='\n'+json.dumps(next(r for r in a.records(bad[key]) if 'kv_layout' in r))
            with self.assertRaisesRegex(ValueError,'exactly one'):a.core(case['bench'],bad,run)
            bad={k:'\n'.join(json.dumps(r) for r in a.records(v) if 'kv_check' not in r)+'\n' for k,v in logs.items()}
            result=a.core(case['bench'],bad,run)
            self.assertTrue(result['problems'])

    def test_performance_keeps_geometry_without_pattern_checks(self):
        case=self.cases[2]
        with tempfile.TemporaryDirectory() as tmp:
            model=Path(tmp);(model/'config.json').write_text(json.dumps(case['model_config']))
            run=case_run(case,model);run.kv_check=False
            logs={}
            for key,text in logs_of(case).items():
                rows=a.records(text)
                for row in rows:
                    for group in row.get('kv_layout',{}).get('groups',[]):
                        group.update(global_heads=None,pattern='rank-v0')
                logs[key]='\n'.join(json.dumps(r) for r in rows if 'kv_check' not in r)+'\n'
            result=a.core(case['bench'],logs,run)
            self.assertEqual(result['problems'],[])
            self.assertEqual(result['bytes']['physical_payload_bytes'],3019898880)
            self.assertEqual(result['bytes']['head_identity_sources'],['derived_from_model_config_and_emitted_page_geometry'])
            broken={k:'\n'.join(json.dumps(r) for r in a.records(v) if 'kv_transfers' not in r)+'\n' for k,v in logs.items()}
            self.assertTrue(a.core(case['bench'],broken,run)['problems'])


    def test_token_tail_padding_and_backend_guard_are_distinct(self):
        case=self.cases[2]
        with tempfile.TemporaryDirectory() as tmp:
            model=Path(tmp);(model/'config.json').write_text(json.dumps(case['model_config']))
            run=case_run(case,model);logs=logs_of(case)
            shortened={}
            for key,text in logs.items():
                rows=a.records(text)
                for row in rows:
                    for request in (row.get('step') or {}).get('requests',[]):
                        request['num_prompt_tokens']=511
                shortened[key]=''.join(json.dumps(r)+'\n' for r in rows)
            result=a.classify(shortened,run,3019898880)
            self.assertEqual(result['transferred_padding_bytes'],40*147456)
            run.server.nixl_backend='mooncake'
            result=a.core(case['bench'],logs,run)
            self.assertTrue(any('guard disagree' in p for p in result['problems']))


class RegenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        synthetic_capture(self.root,cases()[2])

    def change(self,name,function):
        p=self.root/name;obj=json.loads(p.read_text());function(obj);p.write_text(json.dumps(obj));seal(self.root)

    def test_regeneration_uses_raw_and_ignores_derived_status(self):
        result=a.regenerate(self.root)
        self.assertEqual(result['problems'],[])
        self.assertEqual(result['status'],'passed')
        (self.root/'accounting.json').write_text('{"status":"failed"}')
        self.assertEqual(a.regenerate(self.root),result)

    def test_missing_evidence_cannot_be_zero_or_success(self):
        for name in ('prefill-after.json','decode-after.json','client-ledger.jsonl','proxy-ledger.jsonl','decode-input-hashes.json','decode-pod-after.json','inputs/model.tar'):
            with self.subTest(name=name):
                path=self.root/name;content=path.read_bytes();path.unlink();seal(self.root)
                self.assertEqual(a.regenerate(self.root)['status'],'failed')
                path.write_bytes(content);seal(self.root)

    def test_replaced_model_or_raw_artifact_fails(self):
        (self.root/'inputs/model/config.json').write_text('{}')
        self.assertTrue(any('hash' in p for p in a.regenerate(self.root)['problems']))

    def test_missing_read_and_missing_lifecycle_release_fail(self):
        for kind in ('reads_posted','reads_done','finished'):
            name='decode-after.json';original=(self.root/name).read_text()
            def remove(obj):
                rows=a.records(obj['ranks']['dp0.tp0']['steps']['value'])
                for row in rows:row.get('kv_events',{}).pop(kind,None)
                obj['ranks']['dp0.tp0']['steps']['value']=''.join(json.dumps(r)+'\n' for r in rows)
            self.change(name,remove)
            self.assertEqual(a.regenerate(self.root)['status'],'failed',kind)
            (self.root/name).write_text(original);seal(self.root)


    def test_performance_cannot_accept_partial_prefill_or_local_decode_fallback(self):
        self.change('manifest.json',lambda m:m['options'].update(kv_check=False))
        for side in ('prefill','decode'):
            for phase in ('before','after'):
                def unpatterned(obj):
                    rows=a.records(obj['ranks']['dp0.tp0']['steps']['value'])
                    for row in rows:
                        for group in row.get('kv_layout',{}).get('groups',[]):
                            group.update(global_heads=None,pattern='rank-v0')
                    obj['ranks']['dp0.tp0']['steps']['value']=''.join(json.dumps(r)+'\n' for r in rows if 'kv_check' not in r)
                self.change(f'{side}-{phase}.json',unpatterned)
        self.assertEqual(a.regenerate(self.root)['problems'],[])
        for role in ('prefill','decode'):
            original=(self.root/f'{role}-after.json').read_text()
            def partial(obj):
                rows=a.records(obj['ranks']['dp0.tp0']['steps']['value'])
                for row in rows:
                    for request in (row.get('step') or {}).get('requests',[]):
                        request['num_computed_tokens']=0
                        request['num_scheduled_tokens']=1
                obj['ranks']['dp0.tp0']['steps']['value']=''.join(json.dumps(r)+'\n' for r in rows)
            self.change(f'{role}-after.json',partial)
            self.assertEqual(a.regenerate(self.root)['status'],'failed')
            (self.root/f'{role}-after.json').write_text(original);seal(self.root)

    def test_epoch_reset_and_local_boundary_fail(self):
        original=(self.root/'decode-after.json').read_text()
        self.change('decode-after.json',lambda x:x.update(boot_id='reboot'))
        self.assertEqual(a.regenerate(self.root)['status'],'failed')
        (self.root/'decode-after.json').write_text(original)
        self.change('decode-after.json',lambda x:x.update(started=-1))
        self.assertEqual(a.regenerate(self.root)['status'],'failed')



    def test_each_server_keeps_its_own_clock_origin(self):
        def shift(obj):
            offset=1_000_000
            obj['started']+=offset;obj['finished']+=offset
            for rank in obj['ranks'].values():
                rows=a.records(rank['steps']['value'])
                for row in rows:
                    for field in ('t','t_start','t_deadline','t_scheduled'):
                        if field in row:row[field]+=offset
                rank['steps']['value']=''.join(json.dumps(r)+'\n' for r in rows)
        for phase in ('before','after'):self.change(f'decode-{phase}.json',shift)
        self.assertEqual(a.regenerate(self.root)['problems'],[])

    def test_runtime_image_and_dependency_identity_are_regenerated(self):
        self.change('decode-pod-after.json',lambda x:x.update(image_id='example/sim@sha256:'+'b'*64))
        self.assertTrue(any('image epoch' in p for p in a.regenerate(self.root)['problems']))


    def test_shared_interface_is_counted_once(self):
        result=a.regenerate(self.root)
        self.assertEqual(set(result['interface_bytes']),{'prefill','decode'})
        physical=result['physical_transfers']['bytes']
        self.assertEqual(result['interface_bytes']['decode']['rx_bytes'],physical+10000)
        self.assertEqual(result['physical_transfers']['bytes'],3019898880)

    def test_wrong_client_ownership_and_ambiguous_join_fail(self):
        p=self.root/'proxy-ledger.jsonl';rows=a.records(p.read_text());rows[0]['client_id']='unowned'
        p.write_text(''.join(json.dumps(r)+'\n' for r in rows));seal(self.root)
        self.assertEqual(a.regenerate(self.root)['status'],'failed')

    def test_incomplete_cleanup_does_not_sum_to_zero(self):
        def bad(obj):
            obj['metrics']['value']='vllm:kv_cache_usage_perc{engine="0"} 0\n'
        self.change('decode-after.json',bad)
        self.assertEqual(a.regenerate(self.root)['status'],'failed')


    def test_missing_native_counter_and_lazy_threads(self):
        original=(self.root/'decode-after.json').read_text()
        self.change('decode-after.json',lambda obj:obj['metrics'].update(value='\n'.join(l for l in obj['metrics']['value'].splitlines() if not l.startswith(a.NIXL_COUNTERS[0]))))
        self.assertTrue(any('counter missing' in p for p in a.regenerate(self.root)['problems']))
        (self.root/'decode-after.json').write_text(original)
        for role in ('prefill','decode'):
            before=json.loads((self.root/f'{role}-before.json').read_text())
            after=json.loads((self.root/f'{role}-after.json').read_text())
            old=a.records(before['ranks']['dp0.tp0']['steps']['value'])
            thread=[r for r in old if 'threads' in r]
            startup=''.join(json.dumps(r)+'\n' for r in old if 'threads' not in r)
            full=after['ranks']['dp0.tp0']['steps']['value']
            segment=full[len(before['ranks']['dp0.tp0']['steps']['value']):]
            before['ranks']['dp0.tp0']['steps']['value']=startup
            after['ranks']['dp0.tp0']['steps']['value']=startup+''.join(json.dumps(r)+'\n' for r in thread)+segment
            (self.root/f'{role}-before.json').write_text(json.dumps(before))
            (self.root/f'{role}-after.json').write_text(json.dumps(after))
        seal(self.root)
        self.assertEqual(a.regenerate(self.root)['problems'],[])


class CounterTests(unittest.TestCase):
    def test_labelled_counter_delta_and_epoch_reset(self):
        before='bytes_total{engine="0"} 10\nbytes_total{engine="1"} 20\n'
        after='bytes_total{engine="0"} 15\nbytes_total{engine="1"} 27\n'
        self.assertEqual(a.counter_delta(before,after,'bytes_total'),12)
        with self.assertRaisesRegex(ValueError,'reset'):a.counter_delta(after,before,'bytes_total')
        mixed='bytes_total{engine="0"} 5\nbytes_total{engine="1"} 200\n'
        with self.assertRaisesRegex(ValueError,'reset'):a.counter_delta(before,mixed,'bytes_total')
        with self.assertRaisesRegex(ValueError,'missing'):a.counter_delta(before,'bytes_total{engine="0"} 15\n','bytes_total')
        with self.assertRaises(ValueError):a.counter_delta('gauge 1\n','gauge 1\n','missing')

    def test_every_engine_requires_explicit_zero(self):
        text='\n'.join(f'{name}{{engine="{dp}"}} 0' for name in ('vllm:kv_cache_usage_perc','vllm:num_requests_running','vllm:num_requests_waiting') for dp in range(2))
        a.cleanup_metrics(text,2)
        for bad in (text.replace('engine="1"','engine="2"'),text.replace(' 0',' -1',1),text.replace(' 0',' 1',1),text.splitlines()[0]):
            with self.assertRaises(ValueError):a.cleanup_metrics(bad,2)



    def test_tent_baseline_needs_registered_pool_not_an_unrelated_series(self):
        a.registered_pool('tent_registered_buffer_bytes 1024\n',1024)
        for text in ('up 1\n','tent_registered_buffer_bytes 512\n',''):
            with self.assertRaises(ValueError):a.registered_pool(text,1024)

    def test_mooncake_counters_require_receiver_failure_evidence(self):
        text=''.join(f'{name}{{engine="0"}} 0\n' for name in a.MOONCAKE_COUNTERS)
        result=a.engine_deltas(text,text,1,a.MOONCAKE_COUNTERS)
        self.assertEqual(result[a.MOONCAKE_COUNTERS[0]],0)
        bad=text.replace('vllm:mooncake_num_failed_recvs_total{engine="0"} 0',
                         'vllm:mooncake_num_failed_recvs_total{engine="0"} 1')
        with self.assertRaisesRegex(ValueError,'failure'):
            a.engine_deltas(text,bad,1,a.MOONCAKE_COUNTERS)


class LedgerTests(unittest.TestCase):
    def test_client_and_proxy_correlate_concurrent_ids(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                path=str(Path(tmp)/'proxy.jsonl')
                gate=asyncio.Event();seen=[]
                async def app(scope,receive,send):
                    await gate.wait();seen.append(str(wrapper.uuid4()))
                    await send({'type':'http.response.start','status':200})
                wrapper=ledger.ProxyLedger(app,path)
                async def noop(*args):pass
                tasks=[asyncio.create_task(wrapper({'type':'http','path':'/v1/completions','headers':[(b'x-request-id',cid.encode())]},noop,noop)) for cid in ('a','b')]
                gate.set();await asyncio.gather(*tasks)
                rows=a.records(Path(path).read_text())
                self.assertEqual({r['client_id'] for r in rows if r['event']=='mapped'},{'a','b'})
                self.assertEqual(len(set(seen)),2)
        asyncio.run(run())

    def test_client_hook_preserves_timing_and_supplies_header_identity(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                async def original(request_func_input,session,pbar):
                    self.assertTrue(request_func_input.request_id.startswith('kcb-'))
                    return SimpleNamespace(start_time=10.,latency=.5,success=True,error='',prompt_len=256,output_tokens=16)
                path=str(Path(tmp)/'client.jsonl')
                output=await ledger.client_wrapper(original,path)(SimpleNamespace(prompt_len=256,output_len=16,prompt='synthetic',extra_body=None,multi_modal_content=None,chat_messages=None),None)
                bench={'start_times':[10.],'latencies':[.5],'input_lens':[256],'output_lens':[16],'errors':[''],'ttfts':[.1]}
                self.assertEqual(ledger.join_client(bench,a.records(Path(path).read_text()))[0]['latency'],output.latency)
                bench['start_times']=[10.,10.]
                with self.assertRaisesRegex(ValueError,'ambiguous'):ledger.join_client(bench,a.records(Path(path).read_text()))
        asyncio.run(run())




class InputRetentionTests(unittest.TestCase):
    def test_retained_capture_and_pod_hashes_describe_exact_uploaded_bytes(self):
        from src.benchmark.contracts import BenchmarkConfig
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model=root/'model';model.mkdir()
            (model/'config.json').write_text('{}')
            dataset=root/'dataset.jsonl';dataset.write_bytes(b'captured dataset\n')
            timing=root/'timing.jsonl';timing.write_bytes(b'{"device_s":0.1}\n')
            cfg=BenchmarkConfig(backend='vllm',test_id='inputs',initiator_node='a',target_node='b',namespace='n',
                                image='example/sim@sha256:'+'a'*64,log_directory=root,
                                parameters={'model':'org/model','model_revision':'b'*40,'model_path':str(model),
                                            'dataset_name':'custom','dataset_path':str(dataset),'timing':'table','timing_args':{'path':str(timing)}})
            kubectl=Mock()
            kubectl.run.side_effect=lambda args,**kwargs:SimpleNamespace(stdout=args[-1])
            names,receipt=inputs.stage(cfg,kubectl,{'target':'p','initiator':'d'},root/'artifacts')
            saved=root/'artifacts'
            self.assertEqual((saved/'dataset').read_bytes(),dataset.read_bytes())
            self.assertEqual((saved/'timing.jsonl').read_bytes(),timing.read_bytes())
            self.assertEqual((saved/'model/config.json').read_bytes(),b'{}')
            self.assertEqual(inputs.sha256(saved/'model.tar'),receipt['model']['archive_sha256'])
            self.assertEqual(receipt['pod_hashes']['target'][names['dataset_path']],receipt['dataset']['sha256'])
            self.assertEqual(len(kubectl.run.call_args_list),5)

    def test_wrong_pod_hash_fails_and_preserves_original_capture(self):
        from src.benchmark.contracts import BenchmarkConfig
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model=root/'model';model.mkdir();(model/'config.json').write_text('{}')
            cfg=BenchmarkConfig(backend='vllm',test_id='inputs',initiator_node='a',target_node='b',namespace='n',
                                image='example/sim@sha256:'+'a'*64,log_directory=root,
                                parameters={'model':'org/model','model_revision':'b'*40,'model_path':str(model),'dataset_name':'random'})
            kubectl=Mock();kubectl.run.return_value=SimpleNamespace(stdout='{"/model/config.json":"wrong"}')
            with self.assertRaisesRegex(ValueError,'hashes differ'):
                inputs.stage(cfg,kubectl,{'target':'p','initiator':'d'},root/'artifacts')
            self.assertTrue((root/'artifacts/model.tar').is_file())
            receipt=json.loads((root/'artifacts/receipt.json').read_text())
            self.assertEqual(receipt['pod_hashes']['target']['/model/config.json'],'wrong')


class LifecycleTests(unittest.TestCase):
    def test_runtime_digest_is_checked_independently_of_requested_image(self):
        from src.backends.vllm.lifecycle import pod_identity
        from unittest.mock import Mock
        runner=Mock();cfg=SimpleNamespace(namespace='n',image='registry/sim@sha256:'+'a'*64)
        pod={'status':{'podIP':'10.42.0.2','containerStatuses':[{'ready':True,'imageID':'registry/sim@sha256:'+'b'*64,'containerID':'container-1'}]},
             'metadata':{'uid':'uid'},'spec':{'nodeName':'prefill-node'}}
        runner.kubectl.output.return_value=json.dumps(pod)
        with self.assertRaisesRegex(ValueError,'identity'):pod_identity(runner,cfg,'p')
        pod['status']['containerStatuses'][0]['imageID']='docker-pullable://registry/sim@sha256:'+'a'*64
        runner.kubectl.output.return_value=json.dumps(pod)
        self.assertEqual(pod_identity(runner,cfg,'p')['uid'],'uid')

    def exercise(self, fail_client=False):
        from src.backends.vllm import lifecycle
        from src.backends.vllm.runner import VllmRunner
        from src.benchmark.contracts import BackendResources, BenchmarkConfig
        from src.benchmark.network import NetworkAttachment,SecondaryNetwork
        from unittest.mock import Mock
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);root=Path(tmp.name)
        fixture=root/'fixture';fixture.mkdir();synthetic_capture(fixture,cases()[2])
        dep=root/'deployment';dep.mkdir();(dep/'servers').mkdir();(dep/'manifest.yaml').write_text('mock')
        import shutil,base64
        shutil.copytree(fixture/'inputs',dep/'inputs')
        receipt=json.loads((dep/'inputs/receipt.json').read_text())
        deployment=json.loads((fixture/'deployment.json').read_text())
        (dep/'servers/deployment.json').write_text(json.dumps(deployment))
        opts=json.loads((fixture/'manifest.json').read_text())['options']
        opts={k:v for k,v in opts.items() if k in config.DEFAULTS or k in config.MAPPING_OPTIONS or k=='timing_args'}
        for key in ('scenario','prompt_lengths','output_tokens','repeat','cancel_delays','pause_after_cancel_seconds'): opts.pop(key,None)
        opts.update(model='org/model',model_revision='b'*40,model_path=str(dep/'inputs/model'),dataset_name='random',
                    threads=1,run_number=1,profile_interval_seconds=0)
        cfg=BenchmarkConfig(backend='vllm',test_id='ranked',initiator_node='decode-node',target_node='prefill-node',namespace='n',
                            image='example/sim@sha256:'+'a'*64,log_directory=root,parameters=opts,
                            network=SecondaryNetwork(NetworkAttachment('decode-network'),NetworkAttachment('prefill-network')))
        proxy=root/'proxy.py';proxy.write_text('# mock native proxy\n')
        runner=VllmRunner(kubectl=Mock(),proxy_source=proxy)
        def exec_(cfg,pod,command,timeout=None):
            if '/bench/evidence.py' in command:
                if 'hashes' in command:return command[-1]
                if 'stop-client' in command:return '{"stopped":true,"remaining":[]}'
                if 'files' in command:
                    files={'last-client.json':(fixture/'client.json').read_bytes(),
                           'raw/bench.json':json.dumps(cases()[2]['bench']).encode(),
                           'raw/client-ledger.jsonl':(fixture/'client-ledger.jsonl').read_bytes(),'raw/bench.log':b'partial log retained\n'}
                    return json.dumps({n:base64.b64encode(b).decode() for n,b in files.items()})
            if command[0]=='sh':return ''
            if command[0]=='cat':return (fixture/'proxy-ledger.jsonl').read_text()
            if 'connections' in command:return '[]'
            if 'bench' in command:
                if fail_client:raise RuntimeError('client exec failed after saving files')
                return (fixture/'client.json').read_text()
            raise AssertionError(command)
        runner._exec=Mock(side_effect=exec_)
        runner._stop_profilers=Mock(return_value={'prefill':{'remaining':[]},'decode':{'remaining':[]}})
        def capture(runner,cfg,pod,role,path):
            phase='before' if path.name.endswith('-before.json') else 'after'
            value=json.loads((fixture/f'{role}-{phase}.json').read_text());path.write_text(json.dumps(value));return value
        class Sampler:
            def __init__(self,*args,**kwargs):self.cleanups=[]
            def __enter__(self):self.cleanups.append({'target':'','initiator':''});return self
            def __exit__(self,*args):self.cleanups.append({'target':'','initiator':''})
        resources=BackendResources(names={'target':'p','initiator':'d','target_address':'192.0.2.1','initiator_address':'192.0.2.2'},manifest_path=dep/'manifest.yaml')
        with patch.object(lifecycle,'pod_identity',side_effect=lambda runner,cfg,pod:deployment['pods']['target' if pod=='p' else 'initiator']),patch.object(lifecycle,'capture',side_effect=capture),patch.object(lifecycle,'CpuSampler',Sampler):
            if fail_client:
                with self.assertRaisesRegex(ValueError,'evidence failed'):runner.execute(cfg,resources,root/'raw')
                command=root/'raw/ranked-t1-run1-command.json'
            else:command=runner.execute(cfg,resources,root/'raw')
        return root,command

    def test_dataset_actual_execute_path_regenerates_and_parses(self):
        from src.backends.vllm.parser import VllmParser
        root,command=self.exercise()
        metadata=json.loads(command.read_text())
        self.assertEqual(metadata['status'],'passed')
        self.assertEqual(metadata['accounting']['requests'],40)
        observations=VllmParser().parse(root/'raw')
        self.assertEqual(len(observations),40)
        self.assertTrue(all(o.metadata['timing_model_dependent'] for o in observations))
        with self.assertRaises(FileExistsError):
            # Fresh directory contract prevents overwriting an earlier attempt.
            (root/'raw/ranked-t1-run1').mkdir(exist_ok=False)

    def test_failed_exec_keeps_both_rank_snapshots_and_partial_client_files(self):
        root,command=self.exercise(fail_client=True)
        self.assertEqual(json.loads(command.read_text())['status'],'failed')
        evidence=root/'raw/ranked-t1-run1'
        self.assertTrue((evidence/'prefill-after.json').is_file())
        self.assertTrue((evidence/'decode-after.json').is_file())
        self.assertEqual((evidence/'client-raw/raw/bench.log').read_text(),'partial log retained\n')
        self.assertEqual(a.regenerate(evidence)['status'],'failed')


    def test_resource_deletion_failure_is_recorded_and_reported(self):
        from src.backends.vllm import lifecycle
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            runner=Mock();runner._exec.return_value='{"stopped":true,"remaining":[]}'
            runner.kubectl.delete_manifest.return_value=False
            cfg=SimpleNamespace(namespace='n',keep_resources=False)
            resources=SimpleNamespace(names={'target':'p','initiator':'d'},manifest_path=root/'manifest.yaml')
            with self.assertRaisesRegex(ValueError,'deletion failed'):
                lifecycle.cleanup(runner,cfg,resources)
            record=json.loads((root/'servers/resource-cleanup.json').read_text())
            self.assertFalse(record['manifest_deleted'])


if __name__ == "__main__":
    unittest.main()
