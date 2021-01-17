import argparse
import json
import logging
import os
import threading
import uuid
from math import log
from time import time

from aws.AWSConfig import AwsConfig
from aws.ec2Manager import EC2Manager
from aws.s3Manager import S3Manager


def get_instance_configs(instance_ips, extra={}):
    port = AwsConfig.MPC_CONFIG.PORT
    num_faulty_nodes = AwsConfig.MPC_CONFIG.NUM_FAULTY_NODES
    instance_configs = [None] * len(instance_ips)

    for my_id in range(len(instance_ips)):
        config = {
            "N": AwsConfig.MPC_CONFIG.N,
            "t": AwsConfig.MPC_CONFIG.T,
            "my_id": my_id,
            "peers": [f"{ip}:{port}" for ip in instance_ips],
            "reconstruction": {"induce_faults": False},
            "skip_preprocessing": True,
            "extra": extra,
        }

        if num_faulty_nodes > 0:
            num_faulty_nodes -= 1
            config["reconstruction"]["induce_faults"] = True
        instance_configs[my_id] = (my_id, json.dumps(config))

    return instance_configs


def run_commands_on_instances(
    ec2manager, commands_per_instance_list, verbose=True, output_file_prefix=None
):

    node_threads = [
        threading.Thread(
            target=ec2manager.execute_command_on_instance,
            args=[id, commands, verbose, output_file_prefix],
        )
        for id, commands in commands_per_instance_list
    ]

    for thread in node_threads:
        thread.start()
    for thread in node_threads:
        thread.join()

def build_file_name_triple(n, t, i):
    return "sharedata/triples_" + str(n) + '_' + str(t) + '-' + str(i) + '.share'
def build_file_name_rand(n, t, i):
    return "sharedata/rands_" + str(n) + '_' + str(t) + '-' + str(i) + '.share'
def build_file_name_bit(n, t, i):
    return "sharedata/bits_" + str(n) + '_' + str(t) + '-' + str(i) + '.share'
def build_file_name_zero(n, t, i):
    return "sharedata/zeros_" + str(n) + '_' + str(t) + '-' + str(i) + '.share'


def get_dkg_setup_commands(s3manager, instance_ids):
    from honeybadgermpc.preprocessing import PreProcessedElements
    from honeybadgermpc.preprocessing import PreProcessingConstants as Constants

    n, t = AwsConfig.TOTAL_VM_COUNT, AwsConfig.MPC_CONFIG.T


    logging.info("Starting to create preprocessing files.")
    stime = time()
    pp_elements = PreProcessedElements()
    pp_elements.generate_triples(600, n, t)
    pp_elements.generate_rands(600, n, t)
    logging.info(f"Preprocessing files created in {time()-stime}")

    setup_commands = []
    total_time = 0
    logging.info(f"Uploading input files to AWS S3.")
    stime = time()

    triple_urls = s3manager.upload_files(
        [
            build_file_name_triple(n, t, i)
            for i in range(n)
        ]
    )
    input_urls = s3manager.upload_files(
        [
            build_file_name_rand(n, t, i)
            for i in range(n)
        ]
    )
    logging.info(f"Inputs successfully uploaded in {time()-stime} seconds.")

    setup_commands = [
        [
            instance_id,
            [
                "sudo docker pull %s" % (AwsConfig.DOCKER_IMAGE_PATH),
                "mkdir -p sharedata",
                "cd sharedata; curl -sSO %s" % (triple_urls[i]),
                "cd sharedata; curl -sSO %s" % (input_urls[i]),
                "mkdir -p benchmark-logs",
            ],
        ]
        for i, instance_id in enumerate(instance_ids)
    ]
    return setup_commands

def get_bit_dec_setup_commands(s3manager, instance_ids):
    from honeybadgermpc.preprocessing import PreProcessedElements
    from honeybadgermpc.preprocessing import PreProcessingConstants as Constants

    n, t = AwsConfig.TOTAL_VM_COUNT, AwsConfig.MPC_CONFIG.T


    logging.info("Starting to create preprocessing files.")
    stime = time()
    pp_elements = PreProcessedElements()
    pp_elements.generate_triples(150000, n, t)
    pp_elements.generate_rands(66000, n, t)
    pp_elements.generate_bits(10000, n, t)
    pp_elements.generate_zeros(200, n, t)
    logging.info(f"Preprocessing files created in {time()-stime}")

    setup_commands = []
    total_time = 0
    logging.info(f"Uploading input files to AWS S3.")
    stime = time()

    triple_urls = s3manager.upload_files(
        [
            build_file_name_triple(n, t, i)
            for i in range(n)
        ]
    )
    rands_urls = s3manager.upload_files(
        [
            build_file_name_rand(n, t, i)
            for i in range(n)
        ]
    )
    zeros_urls = s3manager.upload_files(
        [
            build_file_name_zero(n, t, i)
            for i in range(n)
        ]
    )
    bits_urls = s3manager.upload_files(
        [
            build_file_name_bit(n, t, i)
            for i in range(n)
        ]
    )
    logging.info(f"Inputs successfully uploaded in {time()-stime} seconds.")

    setup_commands = [
        [
            instance_id,
            [
                "sudo docker pull %s" % (AwsConfig.DOCKER_IMAGE_PATH),
                "mkdir -p sharedata",
                "cd sharedata; curl -sSO %s" % (triple_urls[i]),
                "cd sharedata; curl -sSO %s" % (rands_urls[i]),
                "cd sharedata; curl -sSO %s" % (zeros_urls[i]),
                "cd sharedata; curl -sSO %s" % (bits_urls[i]),
                "mkdir -p benchmark-logs",
            ],
        ]
        for i, instance_id in enumerate(instance_ids)
    ]
    return setup_commands


def trigger_run(run_id, skip_setup, max_k, only_setup, cleanup):
    os.makedirs("sharedata/", exist_ok=True)
    logging.info(f"Run Id: {run_id}")
    ec2manager, s3manager = EC2Manager(), S3Manager(run_id)
    instance_ids, instance_ips = ec2manager.create_instances()

    if cleanup:
        instance_commands = [
            [instance_id, ["sudo docker kill $(sudo docker ps -q); rm -rf *"]]
            for i, instance_id in enumerate(instance_ids)
        ]
        run_commands_on_instances(ec2manager, instance_commands)
        return

    port = AwsConfig.MPC_CONFIG.PORT


    instance_configs = get_instance_configs(instance_ips)
    

    logging.info(f"Uploading config file to S3 in '{AwsConfig.BUCKET_NAME}' bucket.")

    config_urls = s3manager.upload_configs(instance_configs)
    logging.info("Config file upload complete.")

    logging.info("Triggering config update on instances.")
    config_update_commands = [
        [instance_id, ["mkdir -p config", "cd config; curl -sSO %s" % (config_url)]]
        for config_url, instance_id in zip(config_urls, instance_ids)
    ]
    run_commands_on_instances(ec2manager, config_update_commands, False)
    logging.info("Config update completed successfully.")

    if not skip_setup:

        if AwsConfig.MPC_CONFIG.COMMAND.endswith("dkg"):
            setup_commands = get_dkg_setup_commands(s3manager, instance_ids)
        elif AwsConfig.MPC_CONFIG.COMMAND.endswith("dec"):
            setup_commands = get_bit_dec_setup_commands(s3manager, instance_ids)
        else:
            logging.error("Application not supported to run on AWS.")
            raise SystemError
        logging.info("Triggering setup commands.")
        run_commands_on_instances(ec2manager, setup_commands, False)

    if not only_setup:
        logging.info("Setup commands executed successfully.")
        instance_commands = [
            [
                instance_id,
                [
                    f"sudo docker run\
                -p {port}:{port} \
                -v /home/ubuntu/config:/usr/src/HoneyBadgerMPC/config/ \
                -v /home/ubuntu/sharedata:/usr/src/HoneyBadgerMPC/sharedata/ \
                -v /home/ubuntu/benchmark-logs:/usr/src/HoneyBadgerMPC/benchmark-logs/ \
                {AwsConfig.DOCKER_IMAGE_PATH} \
                {AwsConfig.MPC_CONFIG.COMMAND} -d -f config/config-{i}.json"
                ],
            ]
            for i, instance_id in enumerate(instance_ids)
        ]
        logging.info("Triggering MPC commands.")
        run_commands_on_instances(ec2manager, instance_commands)
        # logging.info("Collecting logs.")
        # log_collection_cmds = [
        #     [id, ["cat benchmark-logs/*.log"]] for id in instance_ids
        # ]
        # os.makedirs(run_id, exist_ok=True)
        # run_commands_on_instances(
        #     ec2manager, log_collection_cmds, True, f"{run_id}/benchmark-logs"
        # )
    s3manager.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs HBMPC code on AWS.")
    parser.add_argument(
        "-s",
        "--skip-setup",
        dest="skip_setup",
        action="store_true",
        help="If this is passed, then the setup commands are skipped.",
    )
    parser.add_argument(
        "-c",
        "--cleanup",
        dest="cleanup",
        action="store_true",
        help="This kills all running containers and deletes all stored files.",
    )
    parser.add_argument(
        "-k",
        "--max-k",
        default=AwsConfig.MPC_CONFIG.K,
        type=int,
        dest="max_k",
        help="Maximum value of k for which the inputs need to be \
        created and uploaded during the setup phase. This value is \
        ignored if --skip-setup is passed. (default: `k` in aws_config.json)",
    )
    parser.add_argument(
        "--only-setup",
        dest="only_setup",
        action="store_true",
        help="If this value is passed, then only the setup phase is run,\
         otherwise both phases are run.",
    )
    parser.add_argument(
        "--run-id",
        dest="run_id",
        nargs="?",
        help="If skip setup is passed, then a previous run_id for the same\
        MPC application needs to be specified to pickup the correct input files.",
    )
    args = parser.parse_args()
    if args.skip_setup and args.only_setup:
        parser.error("--only-setup and --skip-setup are mutually exclusive.")
    if args.skip_setup and not args.run_id:
        parser.error("--run-id needs to be passed with --skip-setup.")
    args.run_id = uuid.uuid4().hex if args.run_id is None else args.run_id
    trigger_run(args.run_id, args.skip_setup, args.max_k, args.only_setup, args.cleanup)
