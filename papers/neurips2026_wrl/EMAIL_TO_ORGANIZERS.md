# Email to WRL 2026 organizers (pre-submission fit check)

**To:** wrl2026organizers@robot-learning.ml  
**Subject:** Scope check — VLA–humanoid systems paper for WRL 2026

---

Dear WRL 2026 Organizers,

Quick scope check before we submit to OpenReview.

We have a short systems paper on a Physical AI *deployment* bottleneck: measured UnifoLM-VLA inference is ~57× too slow for 100 Hz Unitree G1 (29-DoF) control. We keep the VLA as a sparse intent generator and add a cheap Echo State Network bridge for 100 Hz joint commands, with profiling, held-out tracking, and live failure analysis (e.g. contact collapse without a disclosed geometric prior).

We do **not** claim zero-shot or cross-embodiment generalization. We see the closest CFP matches as VLA/generalist policies (deployability), real-world deployment / negative results, and failure analysis.

Is that framing appropriate for WRL 2026?

Thanks,  
Osemudiamen Andrew Ihimekpen  
Prairie View A&M University — CREDIT Center  
oihimekpen@pvamu.edu  

Co-author: Lijun Qian (liqian@pvamu.edu)

---

## Send checklist

- [ ] Confirm both author emails
- [ ] CC co-author / advisor if required
- [ ] Send text-only (attach PDF only if they ask)
- [ ] Register OpenReview for both authors now (do not wait on a reply)
- [ ] If no reply by ~20–21 Aug, submit under the VLA + deployment framing anyway
