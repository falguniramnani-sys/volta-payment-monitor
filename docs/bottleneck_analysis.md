# Payment Pipeline Bottleneck Analysis

## Executive Summary

Analysis of 15,000 transactions over 48 hours across Brazil, Mexico, and Colombia reveals three critical bottlenecks in Volta Commerce's payment pipeline. These bottlenecks primarily affect Colombia's PSE payment method and PSP_Gamma's inconsistent performance during peak hours.

## Top 3 Bottlenecks

### 1. Colombia + PSE + PSP_Gamma (P95: 30,795ms)

The worst-performing combination by far. PSP_Gamma's bimodal latency distribution causes extreme tail latencies when processing PSE bank transfers in Colombia. With a P95 of ~31 seconds and approval rate of only 65.4%, this route is severely degraded. PSE's inherently slow bank-redirect flow (2.5x multiplier) compounds PSP_Gamma's unpredictable performance, creating an unacceptable user experience for ~494 transactions.

**Recommendation:** Route all Colombia PSE traffic to PSP_Alpha, which handles PSE with a P95 of 3,948ms — an 87% reduction in tail latency.

### 2. Colombia + PSE + PSP_Beta (P95: 21,176ms)

PSP_Beta's high baseline latency (~3,500ms median) multiplied by PSE's slow processing creates the second-worst bottleneck. While PSP_Beta has the best approval rate (79.7%) among PSE processors, its P95 of 21 seconds makes it unsuitable as the primary PSE router. This affects ~625 transactions with a median latency of 8,678ms.

**Recommendation:** Use PSP_Alpha as the primary router for Colombia PSE. Reserve PSP_Beta as a fallback only when PSP_Alpha is unavailable, leveraging its high approval rate for retry scenarios.

### 3. PSP_Delta Timeout Rate (7.6% overall, 10.4% on Colombia PSE)

PSP_Delta exhibits a systemic timeout problem with an overall rate of 7.6% — well above the 5% warning threshold. On Colombia PSE specifically, the timeout rate spikes to 10.4%, meaning 1 in 10 transactions fails completely. Combined with its 63.3% approval rate on this route, PSP_Delta wastes significant processing capacity on transactions that will never complete.

**Recommendation:** Remove PSP_Delta from PSE routing entirely. For card payments, implement a 10-second timeout cap with automatic failover to PSP_Alpha. Monitor PSP_Delta's timeout rate weekly and consider reducing its traffic allocation until the root cause is resolved.

## Summary

Optimizing PSP routing for Colombia's PSE method alone could improve P95 latency by 87% and save ~1,640 transactions from excessive wait times. The composite health score model (40% latency, 40% approval, 20% reliability) consistently ranks PSP_Alpha as the optimal router for latency-sensitive routes, while PSP_Beta serves best where approval rate is the priority and latency tolerance is higher.
