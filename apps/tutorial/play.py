"""
hbMPC tutorial 2.
Instructions:
   run this with
```
scripts/launch-tmuxlocal.sh apps/tutorial/hbmpc-tutorial-2.py conf/mpc/local
```
"""
import asyncio
import logging
import random
import time
import json
random.seed(562)
from honeybadgermpc.preprocessing import (
    PreProcessedElements as FakePreProcessedElements,
)
from honeybadgermpc.elliptic_curve import Subgroup
from honeybadgermpc.progs.mixins.share_arithmetic import (
    BeaverMultiply,
    BeaverMultiplyArrays,
    MixinConstants,
)
from honeybadgermpc.progs.mixins.dataflow import ShareArray
from honeybadgermpc.field import GF
import honeybadgermpc.progs.fixedpoint 
import time
from  honeybadgermpc.progs.fixedpoint import FixedPoint
mpc_config = {
    MixinConstants.MultiplyShareArray: BeaverMultiplyArrays(),
    MixinConstants.MultiplyShare: BeaverMultiply(),
}

F = 32  # The precision (binary bits)
"""
This implementation of the library is not completely hiding. This leaks information about the bits used in computation which is determinied by the security parameter Kappa.
In particular, we leak O(1/(2^Kappa)) information theorotic bits per operation on a floating point secret.
"""
KAPPA = 32  # Statistical security parameter
K = 64  # Total number of padding bits ()
p = modulus = Subgroup.BLS12_381
Field = GF(p)


def find_divisor(l):
    left = l[0]
    right = l[1]
    divisor = 2 ** (len(l[0]) + len(l[1]))
    if len(l[0]) % 2 == 1:
        divisor = - divisor
    return Field(1) / Field(divisor)

def offline_batch_ltz(ctx, n):
    rs = []
    rs_msb = []
    for i in range(n):
        r_msb = ctx.preproc.get_bit(ctx)
        r_lsbs = Field(random.randint(1,2 ** (K - 1)))
        r = r_msb * (Field(p) - r_lsbs) + (Field(1) - r_msb) * r_lsbs
        rs.append(r)
        rs_msb.append(r_msb)
    return rs,rs_msb


async def batch_ltz(ctx, virables, precomputed_rs, rs_msb):
    num_of_terms = len(virables)
    result = [0 for _ in range(num_of_terms)]
    virables_share = [i.share for i in virables]
    muls =  await (ctx.ShareArray(virables_share) * ctx.ShareArray(precomputed_rs))

    xr_open = await muls.open()
    for i in range(num_of_terms):
        sign = Field(0)
        if xr_open[i].value >= p/2 :
            sign = Field(1)
        result[i] = rs_msb[i] + sign - 2 * rs_msb[i] * sign
    return result

def decision_tree_offline(ctx, num_of_terms, num_of_virables):
    result = [[] for _ in range(num_of_terms)]
    product = [ctx.field(1) for _ in range(num_of_terms)]

    for i in range(num_of_terms):
        p = ctx.field(1)
        for j in range(num_of_virables):
            a = ctx.field(random.randint(1,50))    
            result[i].append(ctx.Share(a)) 
            product[i] = product[i] * (1/a)

        product[i] = ctx.Share(product[i])
    return result,product


async def batch_decision_tree_eval(ctx, terms, precompute_randoms, product):
    num_of_terms = len(terms)
    num_of_virables = len(terms[0])
    combined_terms =  [x for j in terms for x in j]
    combined_precompute_randoms =  [x for j in precompute_randoms for x in j]
    result = [0 for _ in range(num_of_terms)]

    m = await (ctx.ShareArray(combined_terms) * ctx.ShareArray(combined_precompute_randoms))
    open_m = await m.open()
    for i in range(num_of_terms):
        mul = ctx.field(1)
        for j in range(num_of_virables):
            mul = mul * open_m[i * num_of_virables + j]
        result[i] = mul * product[i]
    return result

async def run(ctx, **kwargs):

    # read json from files
    poly = {}
    comparison = {}
    values = {}
    divisors = {}
    poly_j = ''
    comparison_j = ''
    values_j = ""
    with open('/usr/src/HoneyBadgerMPC/apps/tutorial/json_poly.json', 'r') as json_file:
        poly_j =json_file.readline()
    with open('/usr/src/HoneyBadgerMPC/apps/tutorial/json_comparison.json', 'r') as json_file:
        comparison_j = json_file.readline()
    with open('/usr/src/HoneyBadgerMPC/apps/tutorial/json_value.json', 'r') as json_file:
        values_j = json_file.readline()

    poly = json.loads(poly_j)
    comparison = json.loads(comparison_j)
    values = json.loads(values_j)
    for key,value in poly.items():
        divisors[key] = find_divisor(value)
    rs,rs_msb = offline_batch_ltz(ctx, len(comparison))
    # get the number of virables in each poly
    a = random.sample(poly.keys(), 1)  
    b = a[0] 
    precompute_randoms, product = decision_tree_offline(ctx, len(poly), len(poly[b][0]) + len(poly[b][1]) + 1)

    # online phase
    test_input = [FixedPoint(ctx,1), FixedPoint(ctx,0), FixedPoint(ctx,1), FixedPoint(ctx,0), FixedPoint(ctx,1), FixedPoint(ctx,0), FixedPoint(ctx,1), FixedPoint(ctx,0),]
    virables = []
    ids = []
    start =  time.time()
    for node_id, terms in comparison.items():
        ids.append(int(node_id))
        virables.append(FixedPoint(ctx,terms[1]) - test_input[terms[0]])

    comparison_result = await batch_ltz(ctx, virables, rs, rs_msb)
    for i in range(len(comparison_result)):
        comparison_result[i] = (comparison_result[i] - Field(1)/Field(2)) * Field(2)
      
    minus_one_terms = [ (i - Field(1)) for i in comparison_result]
    plus_one_terms = [ (i + Field(1)) for i in comparison_result]

    poly_terms = []
    poly_id = []
    for key,value in poly.items():
        poly_id.append(key)
        terms = []
        for i in value[0]:
            terms.append(minus_one_terms[ids.index(int(i))])
        for i in value[1]:
            terms.append(plus_one_terms[ids.index(int(i))])
        terms.append(values[int(key)])
        poly_terms.append(terms)
    logging.info("start evaluation")
    poly_results = await batch_decision_tree_eval(ctx, poly_terms, precompute_randoms, product)
    logging.info("evaluation finished")
    middle = time.time()
    for i in range(len(poly_results)):
        poly_results[i] = poly_results[i] * divisors[poly_id[i]]
    stop =  time.time()
    logging.info(f"time for division: {stop - middle}")
    logging.info(f"total online time: {stop - start}")
    # open_result = await ctx.ShareArray(poly_results).open() 
    # print(open_result)




    # logging.info("Starting _prog")
    # a = FixedPoint(ctx, 99999999.5)
    # b = FixedPoint(ctx, -3.8)
    # A = await a.open()  # noqa: F841, N806
    # B = await b.open()  # noqa: F841, N806
    # AplusB = await (a + b).open()  # noqa: N806
    # AminusB = await (a - b).open()  # noqa: N806
    # AtimesB = await (await a.__mul__(b)).open()  # noqa: N806
    # logging.info("Starting less than")
    # # AltB = await (await a.lt(b)).open()  # noqa: N806
    # # BltA = await (await b.lt(a)).open()  # noqa: N806
    # AltB = await (await a.new_ltz()).open() 
    # BltA = await (await b.new_ltz()).open() 
    # logging.info("done")
    # logging.info(f"A:{A} B:{B} A-B:{AminusB} A+B:{AplusB}")
    # logging.info(f"A*B:{AtimesB} A<B:{AltB} B<A:{BltA}")
    logging.info("Finished _prog")


async def _run(peers, n, t, my_id, k):
    from honeybadgermpc.ipc import ProcessProgramRunner

    async with ProcessProgramRunner(peers, n, t, my_id, mpc_config) as runner:
        await runner.execute("0", run, k=k)
        bytes_sent = runner.node_communicator.bytes_sent
        print(f"[{my_id}] Total bytes sent out: {bytes_sent}")


if __name__ == "__main__":
    from honeybadgermpc.config import HbmpcConfig
    import sys

    HbmpcConfig.load_config()

    if not HbmpcConfig.peers:
        print(
            f"WARNING: the $CONFIG_PATH environment variable wasn't set. "
            f"Please run this file with `scripts/launch-tmuxlocal.sh "
            f"apps/tutorial/hbmpc-tutorial-2.py conf/mpc/local`"
        )
        sys.exit(1)

    asyncio.set_event_loop(asyncio.new_event_loop())
    loop = asyncio.get_event_loop()
    loop.set_debug(True)
    k = 1000
    try:
#         pp_elements = FakePreProcessedElements()
#         if HbmpcConfig.my_id == 0:
            
#             pp_elements.generate_zeros(20000, HbmpcConfig.N, HbmpcConfig.t)
#             pp_elements.generate_triples(260000, HbmpcConfig.N, HbmpcConfig.t)
#             pp_elements.generate_bits(20000, HbmpcConfig.N, HbmpcConfig.t)
#             pp_elements.preprocessing_done()
#         else:
#             loop.run_until_complete(pp_elements.wait_for_preprocessing())

        loop.run_until_complete(
            _run(HbmpcConfig.peers, HbmpcConfig.N, HbmpcConfig.t, HbmpcConfig.my_id, k)
        )
    finally:
        loop.close()
