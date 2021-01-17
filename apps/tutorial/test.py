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
random.seed(562)
from honeybadgermpc.preprocessing import (
    PreProcessedElements as FakePreProcessedElements,
)
from honeybadgermpc.progs.mixins.share_arithmetic import (
    BeaverMultiply,
    BeaverMultiplyArrays,
    MixinConstants,
)

mpc_config = {
    MixinConstants.MultiplyShareArray: BeaverMultiplyArrays(),
    MixinConstants.MultiplyShare: BeaverMultiply(),
}


def polynomial_offline(ctx, degrees):
    result = []
    product = ctx.field(1)

    for degree in degrees:
        a = ctx.field(random.randint(1,50))
        result.append(ctx.Share(a)) 
        mul_a = a
        for _ in range(degree - 1):
            mul_a = mul_a * a
        product = product * (1/mul_a)

    product = ctx.Share(product)
    return result,product

async def polynomial_term_eval(ctx, terms, degrees, precompute_randoms, product):
    mul = ctx.field(1)
    # m = [0 for _ in range(len(terms))]
    # for i in range(len(terms)):
    #     m[i] = terms[i] * precompute_randoms[i]
    m = await (ctx.ShareArray(terms) * ctx.ShareArray(precompute_randoms))
    open_m = await m.open()
    for i in range(len(terms)):
        res_m = open_m[i]
        for _ in range(degrees[i] - 1):
            res_m = res_m * open_m[i]
        mul = mul * res_m
    result = mul * product
    return result

async def poly(ctx, **kwargs):
    # Computing a dot product by MPC (k openings)

    # generate k random input
    k = kwargs["k"]

    degrees = [random.randint(1,50000) for _ in range(k)]
    terms = [ctx.Share(random.randint(1,50)) for _ in range(k)]

    precompute_randoms,product = polynomial_offline(ctx, degrees)

    start = time.time()
    res = await polynomial_term_eval(ctx, terms, degrees, precompute_randoms, product)
    stop = time.time()
    res_open = await res.open()
    logging.info("online phase time : %s", (stop - start))
    logging.info("%s", res_open)


async def _run(peers, n, t, my_id, k):
    from honeybadgermpc.ipc import ProcessProgramRunner

    async with ProcessProgramRunner(peers, n, t, my_id, mpc_config) as runner:
        await runner.execute("0", poly, k=k)
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
    k = 100
    try:
#         pp_elements = FakePreProcessedElements()
#         if HbmpcConfig.my_id == 0:
            

#             pp_elements.generate_triples(k * 2, HbmpcConfig.N, HbmpcConfig.t)
#             pp_elements.preprocessing_done()
#         else:
#             loop.run_until_complete(pp_elements.wait_for_preprocessing())

        loop.run_until_complete(
            _run(HbmpcConfig.peers, HbmpcConfig.N, HbmpcConfig.t, HbmpcConfig.my_id, k)
        )
    finally:
        loop.close()
