from __future__ import annotations

import json
import unittest
from unittest.mock import Mock

import yaml

from src.benchmark.network import (
    NETWORKS_ANNOTATION,
    NetworkAttachment,
    SecondaryNetwork,
    apply_secondary_network,
    check_requested_address,
    discover_interface_address,
    networks_annotation,
)

MANIFEST = """\
apiVersion: v1
kind: Service
metadata:
  name: svc
---
apiVersion: v1
kind: Pod
metadata:
  name: client
  labels: {app: x}
---
apiVersion: v1
kind: Pod
metadata:
  name: server
"""


def annotation(doc: dict) -> list:
    return json.loads(doc["metadata"]["annotations"][NETWORKS_ANNOTATION])


class NetworkAttachmentTests(unittest.TestCase):
    def test_validation(self) -> None:
        NetworkAttachment("secondary")
        NetworkAttachment("kube-system/secondary", "net2", "fd00::2/64")
        for kwargs in (
            {"attachment": "Bad Name"},
            {"attachment": "a/b/c"},
            {"attachment": "secondary", "interface": "x" * 16},
            {"attachment": "secondary", "ip": "10.0.0.2"},
            {"attachment": "secondary", "ip": "nonsense/24"},
        ):
            with self.assertRaises(ValueError, msg=kwargs):
                NetworkAttachment(**kwargs)

    def test_annotation_includes_namespace_interface_and_requested_ip(self) -> None:
        self.assertEqual(
            json.loads(networks_annotation(
                NetworkAttachment("kube-system/secondary", "net1", "10.0.0.2/24")
            )),
            [{"name": "secondary", "namespace": "kube-system", "interface": "net1",
              "ips": ["10.0.0.2/24"]}],
        )
        self.assertEqual(
            json.loads(networks_annotation(NetworkAttachment("secondary", "net3"))),
            [{"name": "secondary", "interface": "net3"}],
        )

    def test_each_pod_gets_its_own_attachment(self) -> None:
        network = SecondaryNetwork(
            initiator=NetworkAttachment("secondary-a", "net1", "10.0.0.1/24"),
            target=NetworkAttachment("secondary-b", "net2"),
        )
        rendered = apply_secondary_network(
            MANIFEST, network, {"client": "initiator", "server": "target"}
        )
        docs = {d["metadata"]["name"]: d for d in yaml.safe_load_all(rendered)}
        self.assertNotIn("annotations", docs["svc"]["metadata"])
        self.assertEqual(docs["client"]["metadata"]["labels"], {"app": "x"})
        self.assertEqual(
            annotation(docs["client"]),
            [{"name": "secondary-a", "interface": "net1", "ips": ["10.0.0.1/24"]}],
        )
        self.assertEqual(
            annotation(docs["server"]),
            [{"name": "secondary-b", "interface": "net2"}],
        )

    def test_apply_rejects_missing_pod(self) -> None:
        side = NetworkAttachment("secondary")
        with self.assertRaisesRegex(ValueError, "ghost"):
            apply_secondary_network(
                MANIFEST, SecondaryNetwork(side, side), {"ghost": "target"}
            )

    def test_discovery_reads_network_status(self) -> None:
        kubectl = Mock()
        kubectl.output.return_value = json.dumps([
            {"name": "k8s-pod-network", "interface": "eth0", "ips": ["10.244.0.5"]},
            {"name": "ns/secondary", "interface": "net1", "ips": ["192.168.50.7", "fd00::7"]},
        ])
        self.assertEqual(
            discover_interface_address(kubectl, "bench", "pod-a", "net1"), "192.168.50.7"
        )
        jsonpath = kubectl.output.call_args.args[0][-1]
        self.assertIn(r"k8s\.v1\.cni\.cncf\.io/network-status", jsonpath)
        with self.assertRaisesRegex(ValueError, "found: eth0, net1"):
            discover_interface_address(kubectl, "bench", "pod-a", "net9")

    def test_discovery_errors(self) -> None:
        kubectl = Mock()
        kubectl.output.return_value = ""
        with self.assertRaisesRegex(ValueError, "Multus"):
            discover_interface_address(kubectl, "bench", "pod-a", "net1")
        kubectl.output.return_value = json.dumps([{"interface": "net1", "ips": []}])
        with self.assertRaisesRegex(ValueError, "no IP"):
            discover_interface_address(kubectl, "bench", "pod-a", "net1")

    def test_requested_address_must_match(self) -> None:
        requested = NetworkAttachment("secondary", ip="10.0.0.2/24")
        check_requested_address(requested, "pod", "10.0.0.2")
        check_requested_address(NetworkAttachment("secondary"), "pod", "10.9.9.9")
        with self.assertRaisesRegex(ValueError, "requested 10.0.0.2"):
            check_requested_address(requested, "pod", "10.0.0.99")


if __name__ == "__main__":
    unittest.main()
