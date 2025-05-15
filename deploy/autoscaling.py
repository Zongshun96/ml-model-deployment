#!/usr/bin/env python3
import boto3
import time
import os
import base64
import json

# Replace with your actual values
ALB_NAME = "my-alb"
ASG_NAME = "my-asg"
LAUNCH_CONFIG_NAME = "my-launch-config"
TARGET_GROUP_NAME = "my-target-group"
VPC_SUBNETS = ['subnet-0a1f3a70', 'subnet-8b1b96c7', 'subnet-12da3579']  # List of subnet IDs
SECURITY_GROUPS = ['sg-ef62268b', 'sg-0f50940870990528a']
VPC_ID = 'vpc-18af6873'  # Your VPC ID
AMI_ID = 'ami-09fcab8ed06865f2a'  # Your VM image (AMI) ID, e.g., Amazon Linux 2 AMI: 'ami-0fc82f4dabc05670b', Ubuntu22: 'ami-0884d2865dbe9de4b', ms-image: 'ami-086ee73ee039c8623', ss-image: 'ami-09fcab8ed06865f2a', etc.
INSTANCE_TYPE = 'c6i.large'
KEY_NAME = 'experiment-EC2'  # Name of your pre-created key pair
USER_DATA_FILE = "/home/cc/ml-model-deployment/deploy/user_data.run.ms" # Path to your user-data script, e.g., single-stage: user_data.init.ss, user_data.run.ss; multi-stage: user_data.init.ms, user_data.run.ms

# Specify your desired instance profile and role details.
INSTANCE_PROFILE_NAME = "EC2S3AccessProfile"
IAM_ROLE_NAME = "EC2S3AccessRole"
S3_POLICY_ARN = "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"


def read_user_data_script(file_path=USER_DATA_FILE):
    """Read the user-data script from a file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"User data file '{file_path}' not found.")
    with open(file_path, "r") as f:
        return f.read()


def create_instance_profile():
    """
    Create (or reuse) an IAM role and instance profile.
    Returns the instance profile ARN.
    """
    iam_client = boto3.client('iam')

    # Define a trust policy for EC2.
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "ec2.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }
        ]
    }

    # Create the IAM role if needed.
    try:
        iam_client.get_role(RoleName=IAM_ROLE_NAME)
        print(f"IAM role '{IAM_ROLE_NAME}' already exists.")
    except iam_client.exceptions.NoSuchEntityException:
        iam_client.create_role(
            RoleName=IAM_ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="Role for EC2 instances to access S3"
        )
        print(f"Created IAM role: {IAM_ROLE_NAME}")

    # Attach the AmazonS3ReadOnlyAccess policy to the role.
    iam_client.attach_role_policy(
        RoleName=IAM_ROLE_NAME,
        PolicyArn=S3_POLICY_ARN
    )
    print(f"Attached policy {S3_POLICY_ARN} to role {IAM_ROLE_NAME}")

    # Create the instance profile if needed.
    try:
        iam_client.get_instance_profile(InstanceProfileName=INSTANCE_PROFILE_NAME)
        print(f"Instance profile '{INSTANCE_PROFILE_NAME}' already exists.")
    except iam_client.exceptions.NoSuchEntityException:
        iam_client.create_instance_profile(
            InstanceProfileName=INSTANCE_PROFILE_NAME
        )
        print(f"Created instance profile: {INSTANCE_PROFILE_NAME}")
        # Add the role to the instance profile.
        iam_client.add_role_to_instance_profile(
            InstanceProfileName=INSTANCE_PROFILE_NAME,
            RoleName=IAM_ROLE_NAME
        )
        print(f"Added role '{IAM_ROLE_NAME}' to instance profile '{INSTANCE_PROFILE_NAME}'.")

    # It can take a short while for the instance profile to propagate.
    print("Waiting 10 seconds for IAM propagation...")
    time.sleep(10)

    response = iam_client.get_instance_profile(InstanceProfileName=INSTANCE_PROFILE_NAME)
    instance_profile_arn = response["InstanceProfile"]["Arn"]
    print(f"Using instance profile ARN: {instance_profile_arn}")
    return instance_profile_arn

def delete_instance_profile_and_role():
    """
    Clean up the instance profile and IAM role created.
    It removes the role from the instance profile, deletes the instance profile,
    detaches managed policies and deletes any inline policies, and then deletes the role.
    """
    iam_client = boto3.client('iam')

    # Delete the instance profile
    try:
        response = iam_client.get_instance_profile(InstanceProfileName=INSTANCE_PROFILE_NAME)
        roles_in_profile = response['InstanceProfile']['Roles']
        for role in roles_in_profile:
            iam_client.remove_role_from_instance_profile(
                InstanceProfileName=INSTANCE_PROFILE_NAME,
                RoleName=role['RoleName']
            )
            print(f"Removed role {role['RoleName']} from instance profile {INSTANCE_PROFILE_NAME}")
        iam_client.delete_instance_profile(InstanceProfileName=INSTANCE_PROFILE_NAME)
        print(f"Deleted instance profile '{INSTANCE_PROFILE_NAME}'")
    except iam_client.exceptions.NoSuchEntityException:
        print(f"Instance profile '{INSTANCE_PROFILE_NAME}' does not exist.")

    # Delete the IAM role.
    try:
        # List and detach managed policies from the role.
        attached_policies = iam_client.list_attached_role_policies(RoleName=IAM_ROLE_NAME)
        for policy in attached_policies.get('AttachedPolicies', []):
            iam_client.detach_role_policy(
                RoleName=IAM_ROLE_NAME,
                PolicyArn=policy['PolicyArn']
            )
            print(f"Detached policy {policy['PolicyArn']} from role '{IAM_ROLE_NAME}'")
        # Delete any inline policies attached to the role.
        inline_policies = iam_client.list_role_policies(RoleName=IAM_ROLE_NAME)
        for policy_name in inline_policies.get('PolicyNames', []):
            iam_client.delete_role_policy(RoleName=IAM_ROLE_NAME, PolicyName=policy_name)
            print(f"Deleted inline policy '{policy_name}' from role '{IAM_ROLE_NAME}'")
        iam_client.delete_role(RoleName=IAM_ROLE_NAME)
        print(f"Deleted IAM role '{IAM_ROLE_NAME}'")
    except iam_client.exceptions.NoSuchEntityException:
        print(f"IAM role '{IAM_ROLE_NAME}' does not exist.")


def create_alb():
    elbv2 = boto3.client('elbv2')
    response = elbv2.create_load_balancer(
        Name=ALB_NAME,
        Subnets=VPC_SUBNETS,
        SecurityGroups=SECURITY_GROUPS,
        Scheme='internet-facing',
        Type='application',
        IpAddressType='ipv4'
    )
    alb = response['LoadBalancers'][0]
    alb_arn = alb['LoadBalancerArn']
    dns_name = alb['DNSName']
    print(f"Created ALB with ARN: {alb_arn}")
    print(f"ALB DNS Name: {dns_name}")
    print(f"ALB HTTP Address: http://{dns_name}")
    return alb_arn, dns_name


def create_target_group():
    elbv2 = boto3.client('elbv2')
    response = elbv2.create_target_group(
        Name=TARGET_GROUP_NAME,
        Protocol='HTTP',
        Port=5000,
        VpcId=VPC_ID,
        HealthCheckProtocol='HTTP',
        HealthCheckPort='5000',
        HealthCheckPath='/',
        HealthCheckTimeoutSeconds=4,
        HealthCheckIntervalSeconds=5,
        HealthyThresholdCount=2,
        UnhealthyThresholdCount=2,
        TargetType='instance'
    )
    target_group = response['TargetGroups'][0]
    target_group_arn = target_group['TargetGroupArn']
    print(f"Created Target Group with ARN: {target_group_arn}")
    
    # Update the target group's deregistration delay
    modify_response = boto3.client('elbv2').modify_target_group_attributes(
        TargetGroupArn=target_group_arn,
        Attributes=[
            {
                'Key': 'deregistration_delay.timeout_seconds',
                'Value': '10'  # Desired delay in seconds
            }
        ]
    )
    print(modify_response)
    return target_group_arn


def create_listener(alb_arn, target_group_arn):
    elbv2 = boto3.client('elbv2')
    response = elbv2.create_listener(
        LoadBalancerArn=alb_arn,
        Protocol='HTTP',
        Port=5000,
        DefaultActions=[
            {
                'Type': 'forward',
                'TargetGroupArn': target_group_arn
            }
        ]
    )
    listener = response['Listeners'][0]
    listener_arn = listener['ListenerArn']
    print(f"Created Listener with ARN: {listener_arn}")
    return listener_arn


def create_launch_configuration(instance_profile_arn):
    asg_client = boto3.client('autoscaling')
    
    # Read user-data script from file and encode in base64
    user_data_script = read_user_data_script()
    user_data_encoded = base64.b64encode(user_data_script.encode()).decode()
    
    # Create launch configuration with a block device mapping to specify a custom disk size (30 GiB)
    asg_client.create_launch_configuration(
        LaunchConfigurationName=LAUNCH_CONFIG_NAME,
        ImageId=AMI_ID,
        InstanceType=INSTANCE_TYPE,
        KeyName=KEY_NAME,
        UserData=user_data_encoded,
        SecurityGroups=SECURITY_GROUPS,
        IamInstanceProfile=instance_profile_arn,  # Use the ARN here
        BlockDeviceMappings=[
            {
                'DeviceName': '/dev/sda1',  # Adjust as needed for your AMI's root device.
                'Ebs': {
                    'VolumeSize': 20,       # Disk size in GiB.
                    'VolumeType': 'gp3',    # You can use 'gp3', 'io1', etc. if needed.
                    'DeleteOnTermination': True,
                },
            },
        ]
    )
    print(f"Created Launch Configuration: {LAUNCH_CONFIG_NAME}")


def delete_launch_configuration():
    asg_client = boto3.client('autoscaling')
    try:
        asg_client.delete_launch_configuration(
            LaunchConfigurationName=LAUNCH_CONFIG_NAME
        )
        print(f"Deleted Launch Configuration: {LAUNCH_CONFIG_NAME}")
    except Exception as e:
        print(f"Error deleting launch configuration: {e}")


def create_asg(target_group_arn, instance_profile_arn):
    asg_client = boto3.client('autoscaling')
    # Create the launch configuration first
    create_launch_configuration(instance_profile_arn)
    # Create the Auto Scaling Group and associate it with the target group
    asg_client.create_auto_scaling_group(
        AutoScalingGroupName=ASG_NAME,
        LaunchConfigurationName=LAUNCH_CONFIG_NAME,
        MinSize=0,
        MaxSize=5,
        DesiredCapacity=0,
        VPCZoneIdentifier=",".join(VPC_SUBNETS),
        TargetGroupARNs=[target_group_arn]
    )
    print(f"Created Auto Scaling Group: {ASG_NAME}")


def delete_asg():
    asg_client = boto3.client('autoscaling')
    try:
        asg_client.delete_auto_scaling_group(
            AutoScalingGroupName=ASG_NAME,
            ForceDelete=True
        )
        print(f"Deleted Auto Scaling Group: {ASG_NAME}")
    except Exception as e:
        print(f"Error deleting ASG: {e}")
    # Wait a bit to ensure ASG deletion before deleting the launch configuration
    time.sleep(10)
    delete_launch_configuration()


def delete_listener(listener_arn):
    elbv2 = boto3.client('elbv2')
    try:
        elbv2.delete_listener(ListenerArn=listener_arn)
        print(f"Deleted Listener: {listener_arn}")
    except Exception as e:
        print(f"Error deleting listener: {e}")


def delete_target_group(target_group_arn):
    elbv2 = boto3.client('elbv2')
    try:
        elbv2.delete_target_group(TargetGroupArn=target_group_arn)
        print(f"Deleted Target Group: {target_group_arn}")
    except Exception as e:
        print(f"Error deleting target group: {e}")


def delete_alb(alb_arn):
    elbv2 = boto3.client('elbv2')
    try:
        elbv2.delete_load_balancer(LoadBalancerArn=alb_arn)
        print("Deleted ALB")
    except Exception as e:
        print(f"Error deleting ALB: {e}")


if __name__ == '__main__':
    # First, create (or ensure) the IAM role and instance profile exist.
    instance_profile_arn = create_instance_profile()

    # Create ALB, Target Group, and Listener.
    alb_arn, dns_name = create_alb()
    target_group_arn = create_target_group()
    listener_arn = create_listener(alb_arn, target_group_arn)
    
    # Create Auto Scaling Group with instance profile ARN and register to the target group.
    create_asg(target_group_arn, instance_profile_arn)
    
    # # Print the ALB HTTP address.
    # print("\nALB HTTP Address:", f"http://{dns_name}")
    
    # Wait for testing, then clean up resources.
    input("\nPress Enter to delete all resources...")
    
    # Delete ASG, Listener, Target Group, and ALB
    delete_asg()
    delete_listener(listener_arn)
    delete_target_group(target_group_arn)
    delete_alb(alb_arn)

    delete_instance_profile_and_role()