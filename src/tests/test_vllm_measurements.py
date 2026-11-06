"""Offline CPU/window regressions; generated process samples are synthetic."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.backends.vllm import accounting, report
from src.backends.vllm.analyzer import VllmAnalyzer
from src.backends.vllm.parser import StepLog, VllmParser
from src.backends.vllm.profile import CONNECTOR_THREAD, summarize_profile
from src.graphing.cli import main as graph_main, AXIS_LABELS
from src.graphing.contracts import AnalysisRequest, Observation
from src.tests.test_vllm_accounting import cases, seal, synthetic_capture, synthetic_profiles
from src.tests.test_vllm_backend import profile_text


def encode(rows):
    return ''.join(json.dumps(row)+'\n' for row in rows)




class ProfileWindowTests(unittest.TestCase):
    def measure(self,rows=None,**kwargs):
        return summarize_profile(encode(rows) if rows is not None else profile_text(),StepLog(),False,**kwargs)

    def test_window_interpolates_counters_and_excludes_setup(self):
        result=self.measure(window=[0.5,2.5])
        self.assertEqual(result['window_s'],2)
        self.assertEqual(result['cpu']['server']['core_seconds'],3)
        self.assertEqual(result['cpu']['profiler']['core_seconds'],0.04)
        self.assertEqual(result['network']['net1']['tx_bytes'],1500)
        self.assertEqual(result['coverage_fraction'],1)
        self.assertTrue(result['node_cpu_overlaps_process_cpu'])

    def test_sparse_thread_rows_retain_unchanged_counters(self):
        rows=accounting.records(profile_text())
        for row in rows[1:]:
            worker=next(thread for thread in row['th'] if thread[0]==12)
            worker[5]=0
            if row['t'] in (1,3):
                row['th']=[thread for thread in row['th'] if thread[0]!=12]
        result=self.measure(rows,window=[0.5,1.5])
        worker=next(thread for thread in result['threads'] if thread['tid']==12)
        self.assertEqual(worker['core_seconds'],0.5)

    def test_sampling_gap_is_not_extrapolated_as_complete_cpu(self):
        rows=accounting.records(profile_text())
        for row,t in zip(rows[1:],(0,1,5,6)):row['t']=t
        result=self.measure(rows)
        self.assertEqual(result['coverage_s'],2)
        self.assertFalse(result['complete'])
        self.assertIsNone(result['cpu']['server']['core_seconds'])
        self.assertIsNone(result['node_core_seconds'])
        self.assertEqual(result['gaps'],[[1,5]])

    def test_window_clipping_and_missing_process_are_explicit(self):
        result=self.measure(window=[-0.5,2.5],servers={'7':{'start_ticks':1},'8':{'start_ticks':1}})
        self.assertFalse(result['complete'])
        self.assertIsNone(result['cpu']['server']['core_seconds'])
        self.assertGreater(result['cpu']['server']['observed_core_seconds'],0)

    def test_thread_and_process_identity_reuse_never_bridges_counters(self):
        rows=accounting.records(profile_text())
        for row in rows[3:]:
            next(thread for thread in row['th'] if thread[0]==11)[10]=2
            next(process for process in row['ps'] if process[0]==7)[1]=2
        steps=StepLog.parse(json.dumps({'threads':{'11':CONNECTOR_THREAD},'t':0.1}))
        result=summarize_profile(encode(rows),steps,False)
        epochs=[thread for thread in result['threads'] if thread['tid']==11]
        self.assertEqual(len(epochs),2)
        self.assertEqual(sum(thread['core_seconds'] for thread in epochs),2)
        self.assertIsNone(next(thread for thread in epochs if thread['start_ticks']==2)['label'])
        self.assertFalse(result['cpu']['server']['complete'])

    def test_client_proxy_profiler_and_server_cpu_are_separate(self):
        rows=accounting.records(profile_text())
        for row in rows[1:]:
            t=row['t']
            row['ps'] += [[55,1,1,'/bench/ledger.py client /tmp/ledger',50*t,0,'python'],
                          [56,1,1,'/bench/ledger.py proxy nixl',20*t,0,'python']]
        result=self.measure(rows)
        self.assertEqual(result['cpu']['server']['core_seconds'],4)
        self.assertAlmostEqual(result['cpu']['client']['core_seconds'],1.5)
        self.assertAlmostEqual(result['cpu']['proxy']['core_seconds'],0.6)
        self.assertEqual(result['cpu']['profiler']['core_seconds'],0.06)
        self.assertEqual(len(result['cpu']['server']['processes']),1)

    def test_current_connector_label_and_counter_resets(self):
        native=Path('vllm/vllm/v1/worker/simulated/model_runner.py').read_text()
        self.assertTrue(CONNECTOR_THREAD in native)
        rows=accounting.records(profile_text());rows[-1]['cg']=0
        with self.assertRaisesRegex(ValueError,'counter reset'):self.measure(rows)


    def test_throttling_uses_local_window_with_explicit_partial_and_missing_coverage(self):
        rows=accounting.records(profile_text())
        for row in rows[1:]:
            row['throttle']={'nr_periods':100+10*row['t'],'nr_throttled':20+2*row['t'],
                             'throttled_usec':1000+100000*row['t']}
        measured=self.measure(rows,window=[0.5,2.5])['throttling']
        self.assertTrue(measured['complete'])
        self.assertEqual(measured['clock'],'node_monotonic')
        self.assertEqual(measured['local_interval'],[0.5,2.5])
        self.assertEqual(measured['delta'],{'nr_periods':20,'nr_throttled':4,'throttled_usec':200000})
        self.assertEqual(measured['throttled_seconds'],0.2)
        self.assertEqual(measured['throttled_period_fraction'],0.2)
        rows[-1]['throttle']=None
        partial=self.measure(rows,window=[0.5,2.5])['throttling']
        self.assertTrue(partial['available'])
        self.assertFalse(partial['complete'])
        self.assertEqual(partial['coverage_s'],1.5)
        self.assertIsNone(partial['delta'])
        self.assertIsNone(partial['throttled_seconds'])
        missing=self.measure(window=[0.5,2.5])['throttling']
        self.assertFalse(missing['available'])
        self.assertIsNone(missing['observed_delta'])
        self.assertIsNone(missing['throttled_period_fraction'])
        rows[-1]['throttle']={'nr_periods':0,'nr_throttled':0,'throttled_usec':0}
        with self.assertRaisesRegex(ValueError,'counter reset'):self.measure(rows)

    def test_throttling_excludes_sampling_gaps(self):
        rows=accounting.records(profile_text())
        for row,t in zip(rows[1:],(0,1,5,6)):
            row['t']=t
            row['throttle']={'nr_periods':t*10,'nr_throttled':t*2,'throttled_usec':t*100000}
        measured=self.measure(rows)['throttling']
        self.assertEqual(measured['coverage_s'],2)
        self.assertEqual(measured['observed_delta']['throttled_usec'],200000)
        self.assertFalse(measured['complete'])
        self.assertIsNone(measured['throttled_seconds'])


class CpuLimitCaptureTests(unittest.TestCase):
    def test_snapshot_uses_existing_affinity_collector_and_retains_errors(self):
        from src.backends.vllm import evidence, pod_helper
        limits={'cpu_count':8,'cpu.max':'100000 100000','cpuset.cpus.effective':'2-3',
                'processes':[{'pid':10,'command':'vllm serve /model','cpus':'2-3'}]}
        with patch.dict('sys.modules',{'pod_helper':pod_helper}), \
             patch.object(pod_helper,'affinity',return_value=limits) as affinity, \
             patch.object(pod_helper,'interfaces',return_value={}), \
             patch.object(pod_helper,'host_cpu',return_value=[0]*8), \
             patch.object(evidence,'processes',return_value={}), \
             patch.object(evidence.Path,'rglob',return_value=[]), \
             patch.object(evidence.metadata,'version',return_value='test'), \
             patch.object(evidence.subprocess,'check_output',return_value='test'), \
             patch.object(evidence,'urlopen') as fetch:
            fetch.return_value.__enter__.return_value.read.return_value=b''
            snapshot=evidence.snapshot(1,1,False)
            self.assertEqual(snapshot['cpu_limits'],{'ok':True,'value':limits})
            self.assertEqual(snapshot['clock'],'node_monotonic')
            self.assertLessEqual(snapshot['started'],snapshot['finished'])
            affinity.side_effect=OSError('unreadable limits')
            failed=evidence.snapshot(1,1,False)['cpu_limits']
            self.assertFalse(failed['ok'])
            self.assertIn('unreadable limits',failed['error'])

    def test_profiler_retains_native_throttling_counters_and_missing_reads(self):
        import contextlib
        import io
        from src.backends.vllm import pod_helper
        expected={'nr_periods':10,'nr_throttled':3,'throttled_usec':2000}
        with patch.object(pod_helper,'_read',return_value='usage_usec 30\nnr_periods 10\nnr_throttled 3\nthrottled_usec 2000\n'):
            self.assertEqual(pod_helper.cgroup_throttling(),expected)
        for text in ('','usage_usec 30\nnr_periods 10\n'):
            with self.subTest(text=text),patch.object(pod_helper,'_read',return_value=text):
                self.assertIsNone(pod_helper.cgroup_throttling())
        output=io.StringIO()
        with patch.object(pod_helper,'cgroup_throttling',return_value=expected),contextlib.redirect_stdout(output):
            pod_helper.profile(0.01,0.02)
        self.assertTrue(all(row['throttle']==expected for row in accounting.records(output.getvalue())[1:]))


class ReportTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)
        synthetic_capture(self.root,cases()[2]);synthetic_profiles(self.root)

    def change(self,name,function):
        p=self.root/name;value=json.loads(p.read_text());function(value);p.write_text(json.dumps(value));seal(self.root)

    def test_reports_regenerate_counts_denominators_and_idle_separately(self):
        result=report.regenerate(self.root)
        self.assertEqual(result['problems'],[])
        self.assertEqual(result['status'],'complete')
        self.assertEqual(result['requests'],{'attempted':40,'completed':40,'failed':0,'unfinished':0})
        self.assertEqual(len(result['observations']),8)
        for server in result['servers'].values():
            workload=server['windows']['workload']
            self.assertAlmostEqual(server['server_core_seconds'],workload['window_s'],places=5)
            self.assertAlmostEqual(server['core_seconds_per_gib']['physical_payload'],
                                   workload['window_s']/(result['bytes']['physical_payload_bytes']/report.GIB),places=5)
            self.assertAlmostEqual(server['windows']['idle']['cpu']['server']['cores'],1)
            self.assertGreater(server['windows']['drain']['window_s'],0)
        self.assertIn('polling/scheduler observation delay',result['latency']['scheduler_observed_read_wait']['timestamps'])
        (self.root/'report.json').write_text('{"status":"complete","invented":true}')
        (self.root/'accounting.json').write_text('{"status":"failed"}')
        self.assertEqual(report.regenerate(self.root),result)
        self.assertIn('core-seconds/GiB',report.render_text(result))

    def test_independent_server_clock_offsets_preserve_cpu(self):
        original=report.regenerate(self.root)
        offset=1_000_000
        for phase in ('before','after'):
            def shift(snapshot):
                snapshot['started']+=offset;snapshot['finished']+=offset
                for rank in snapshot['ranks'].values():
                    rows=accounting.records(rank['steps']['value'])
                    for row in rows:
                        for field in ('t','t_start','t_deadline','t_scheduled'):
                            if field in row:row[field]+=offset
                    rank['steps']['value']=encode(rows)
            self.change(f'decode-{phase}.json',shift)
        rows=accounting.records((self.root/'decode-profile.jsonl').read_text())
        for row in rows:
            if 't' in row:row['t']+=offset
        (self.root/'decode-profile.jsonl').write_text(encode(rows))
        self.change('manifest.json',lambda m:m['execution_envelopes']['decode'].update(
            {key:value+offset for key,value in m['execution_envelopes']['decode'].items()}))
        result=report.regenerate(self.root)
        self.assertEqual(result['problems'],[])
        self.assertAlmostEqual(result['servers']['decode']['server_core_seconds'],
                               original['servers']['decode']['server_core_seconds'])
        for field,value in result['servers']['decode']['windows']['workload']['throttling']['delta'].items():
            self.assertAlmostEqual(value,original['servers']['decode']['windows']['workload']['throttling']['delta'][field],places=4)

    def test_previous_repetitions_do_not_change_windows_request_populations_or_cpu(self):
        # A label emitted before the run must survive segmentation.
        for role in ('prefill','decode'):
            before=json.loads((self.root/f'{role}-before.json').read_text())
            label=encode([{'threads':{'10':CONNECTOR_THREAD},'t':before['finished']}])
            for phase in ('before','after'):
                def add_label(snapshot):
                    for rank in snapshot['ranks'].values():rank['steps']['value']=label+rank['steps']['value']
                self.change(f'{role}-{phase}.json',add_label)
        original=report.regenerate(self.root)
        for server in original['servers'].values():
            self.assertTrue(server['windows']['workload']['kv_connector']['available'])
        historical='00000000-0000-0000-0000-000000000001'
        for role in ('prefill','decode'):
            before=json.loads((self.root/f'{role}-before.json').read_text())
            after=json.loads((self.root/f'{role}-after.json').read_text())
            for rank,initial in before['ranks'].items():
                old=initial['steps']['value']
                full=after['ranks'][rank]['steps']['value']
                rows=accounting.records(full[len(old):])
                history=[]
                for row in rows:
                    if not any(field in row for field in ('kv_events','step')):continue
                    # Disjoint request IDs and earlier timestamps make any leakage visible.
                    text=json.dumps(row)
                    for request in set(accounting.UUID.findall(text)):text=text.replace(request,historical)
                    row=json.loads(text)
                    for key in ('t','t_start','t_deadline','t_scheduled'):
                        if key in row:row[key]-=1000
                    history.append(row)
                prefix=encode(history)
                initial['steps']['value']=old+prefix
                after['ranks'][rank]['steps']['value']=old+prefix+full[len(old):]
            for phase,snapshot in (('before',before),('after',after)):
                (self.root/f'{role}-{phase}.json').write_text(json.dumps(snapshot))
        seal(self.root)
        self.assertEqual(accounting.regenerate(self.root)['status'],'passed')
        result=report.regenerate(self.root)
        self.assertEqual(result,original)
        self.assertEqual(result['latency']['scheduler_observed_read_wait']['count'],40)

    def test_execution_limits_and_throttling_have_local_windows_and_explicit_unavailability(self):
        result=report.regenerate(self.root)
        self.assertEqual(result['status'],'complete')
        for role,server in result['servers'].items():
            limits=server['execution_limits']
            self.assertTrue(limits['available'])
            self.assertTrue(limits['stable'])
            self.assertEqual(limits['snapshots']['before']['value']['cpu.max'],'100000 100000')
            self.assertEqual(limits['snapshots']['before']['value']['cpuset.cpus.effective'],'0-7')
            workload=server['windows']['workload']
            throttle=workload['throttling']
            self.assertTrue(throttle['complete'])
            self.assertEqual(throttle['clock'],'node_monotonic')
            self.assertEqual(throttle['local_interval'],workload['local_interval'])
            self.assertEqual(throttle['coverage_s'],workload['coverage_s'])
            self.assertAlmostEqual(throttle['throttled_period_fraction'],0.25)
            rows=accounting.records((self.root/f'{role}-profile.jsonl').read_text())
            for row in rows[1:]:row.pop('throttle')
            (self.root/f'{role}-profile.jsonl').write_text(encode(rows))
        # Runner-clock counters cannot replace missing server-local samples.
        for side in ('target','initiator'):
            (self.root/f'cpu-{side}.log').write_text('0 0 0 0 0\n1 1000000 0 10 0\n')
        self.change('decode-after.json',lambda s:s['cpu_limits']['value'].update({'cpu.max':'200000 100000'}))
        result=report.regenerate(self.root)
        self.assertFalse(result['servers']['decode']['execution_limits']['stable'])
        self.assertFalse(result['servers']['decode']['windows']['workload']['throttling']['available'])
        self.assertIsNone(result['servers']['decode']['windows']['workload']['throttling']['throttled_seconds'])
        self.change('decode-before.json',lambda s:s.update(cpu_limits={'ok':False,'error':'unreadable cpu.max'}))
        result=report.regenerate(self.root)
        limits=result['servers']['decode']['execution_limits']
        self.assertFalse(limits['available'])
        self.assertIsNone(limits['stable'])
        self.assertIn('unreadable cpu.max',report.render_text(result))

    def test_failed_final_request_uses_full_execution_envelopes(self):
        rows=accounting.records((self.root/'client-ledger.jsonl').read_text())
        last=max((row for row in rows if row['event']=='finished'),key=lambda r:r['t'])
        last.update(success=False,error='synthetic last request failure',t=last['t']+0.2)
        (self.root/'client-ledger.jsonl').write_text(encode(rows));seal(self.root)
        result=report.regenerate(self.root)
        self.assertEqual(result['status'],'incomplete')
        self.assertEqual(result['requests']['failed'],1)
        self.assertEqual(result['workload']['client_local_interval'][1],last['t'])
        self.assertEqual(result['observations'],[])
        for server in result['servers'].values():
            self.assertIn('including failed/incomplete requests',server['workload_window_source'])

    def test_missing_profile_cannot_produce_measurements(self):
        (self.root/'decode-profile.jsonl').unlink();seal(self.root)
        result=report.regenerate(self.root)
        self.assertEqual(result['status'],'incomplete')
        self.assertEqual(result['observations'],[])
        self.assertTrue(any('decode-profile' in problem for problem in result['problems']))

    def test_zero_or_missing_denominators_never_produce_cost_observations(self):
        verdict=accounting.regenerate(self.root)
        for value in (0,None):
            with self.subTest(value=value):
                missing=deepcopy(verdict);missing['bytes']['useful_kv_bytes']=value
                result=report.regenerate(self.root,missing)
                self.assertEqual(result['status'],'incomplete')
                self.assertEqual(result['observations'],[])
                self.assertIsNone(result['servers']['decode']['core_seconds_per_gib']['useful_kv'])

    def test_missing_transfer_duration_fails_and_unknown_post_or_wait_is_explicit(self):
        verdict=accounting.regenerate(self.root)
        missing=deepcopy(verdict);missing['operations'][0]['entry'][1]=None
        result=report.regenerate(self.root,missing)
        self.assertEqual(result['status'],'incomplete')
        self.assertEqual(result['observations'],[])
        self.assertTrue(any('required connector transfer timing' in problem for problem in result['problems']))
        optional=deepcopy(verdict);optional['operations'][0]['entry'][2]=None
        result=report.regenerate(self.root,optional)
        self.assertEqual(result['status'],'complete')
        self.assertFalse(result['latency']['connector_post_or_wait']['available'])

    def test_tp_rank_aggregation_does_not_duplicate_shared_process_cpu(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);synthetic_capture(root,cases()[3]);synthetic_profiles(root)
            path=root/'decode-after.json';snapshot=json.loads(path.read_text())
            rows=accounting.records(snapshot['ranks']['dp0.tp3']['steps']['value'])
            final=next(row for row in reversed(rows) if row.get('kv_events',{}).get('finished'))
            final['t_scheduled']+=0.1;final['t']+=0.1
            snapshot['ranks']['dp0.tp3']['steps']['value']=encode(rows)
            path.write_text(json.dumps(snapshot));seal(root)
            result=report.regenerate(root)
            self.assertEqual(result['problems'],[])
            self.assertEqual(result['requests']['completed'],40)
            self.assertEqual(len(result['servers']),2)
            for server in result['servers'].values():
                measured=server['windows']['workload']
                self.assertAlmostEqual(measured['cpu']['server']['cores'],1,places=5)
                self.assertEqual(len(measured['cpu']['server']['processes']),1)
            self.assertEqual(result['servers']['decode']['windows']['workload']['local_interval'][1],final['t_scheduled'])

    def test_sampling_gaps_reject_a_successful_measurement(self):
        rows=accounting.records((self.root/'decode-profile.jsonl').read_text())
        rows=rows[:2]+rows[-2:]
        (self.root/'decode-profile.jsonl').write_text(encode(rows));seal(self.root)
        result=report.regenerate(self.root)
        self.assertEqual(result['status'],'incomplete')
        self.assertIsNone(result['servers']['decode']['server_core_seconds'])
        self.assertEqual(result['observations'],[])

    def test_profile_cannot_be_swapped_between_server_boot_epochs(self):
        rows=accounting.records((self.root/'decode-profile.jsonl').read_text())
        rows[0]['boot_id']='another-server-boot'
        (self.root/'decode-profile.jsonl').write_text(encode(rows));seal(self.root)
        result=report.regenerate(self.root)
        self.assertEqual(result['status'],'incomplete')
        self.assertEqual(result['observations'],[])
        self.assertTrue(any('boot identity' in problem for problem in result['problems']))

    def test_plots_use_one_run_point_per_role_and_correct_units(self):
        # Real parser/report/CLI path; fabricated CPU counters remain test-only.
        parent=self.root/'raw';parent.mkdir();command=parent/'measure-command.json'
        command.write_text(json.dumps({'evidence_directory':'..','parameters':{'threads':1}}))
        observations=VllmParser().parse(parent)
        cpu=[point for point in observations if point.metric=='cpu']
        self.assertEqual(len(cpu),2)
        points=VllmAnalyzer().aggregate(tuple(observations),AnalysisRequest(metric='cpu',x_dimension='threads',series_dimensions=('side',)))
        self.assertEqual(len(points),2)
        self.assertAlmostEqual(points[0].y_value,1)
        self.assertEqual({point.unit for point in cpu},{'cores'})
        for metric in ('cpu','useful-kv-throughput','cpu-per-gib-physical','cpu-per-gib-useful'):
            with self.subTest(metric=metric),patch('src.graphing.cli.load_observations',return_value=tuple(observations)), \
                 patch('src.graphing.cli.render_graph') as render:
                self.assertEqual(graph_main(['--backend','vllm','--name','measure','--dir',str(self.root),
                                             '--x','threads','--y',metric]),0)
                if metric!='cpu':self.assertEqual(render.call_args.args[1].y_label,AXIS_LABELS[metric])
        malformed=Observation(metric='cpu',value=1,unit='core-s/GiB',dimensions={'threads':1,'side':'prefill'},sample_kind='run',run_id='other')
        with self.assertRaisesRegex(ValueError,'mixes measurement units'):
            VllmAnalyzer().aggregate((*cpu,malformed),AnalysisRequest(metric='cpu',x_dimension='threads',series_dimensions=('side',)))


if __name__=='__main__':
    unittest.main()
