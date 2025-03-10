#!/usr/bin/env python3
import boto3
import time

# Replace with your actual values
ALB_NAME = "my-alb"
ASG_NAME = "my-asg"
LAUNCH_CONFIG_NAME = "my-launch-config"
TARGET_GROUP_NAME = "my-target-group"
VPC_SUBNETS = ['subnet-0a1f3a70', 'subnet-8b1b96c7', 'subnet-12da3579']  # List of subnet IDs
SECURITY_GROUPS = ['sg-ef62268b', 'sg-0f50940870990528a']
VPC_ID = 'vpc-18af6873'  # Your VPC ID
AMI_ID = 'ami-0fc82f4dabc05670b'  # Your VM image (AMI) ID, e.g., Amazon Linux 2 AMI
INSTANCE_TYPE = 't2.micro'
KEY_NAME = 'experiment-EC2'  # Name of your pre-created key pair
USER_DATA_FILE = "user_data.sh"  # Path to the user-data file

def read_user_data_script(file_path=USER_DATA_FILE):
    """Read the user-data script from a file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"User data file '{file_path}' not found.")
    with open(file_path, "r") as f:
        return f.read()
    
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
        Port=80,
        VpcId=VPC_ID,
        HealthCheckProtocol='HTTP',
        HealthCheckPort='80',
        HealthCheckPath='/',
        TargetType='instance'
    )
    target_group = response['TargetGroups'][0]
    target_group_arn = target_group['TargetGroupArn']
    print(f"Created Target Group with ARN: {target_group_arn}")
    return target_group_arn

def create_listener(alb_arn, target_group_arn):
    elbv2 = boto3.client('elbv2')
    response = elbv2.create_listener(
        LoadBalancerArn=alb_arn,
        Protocol='HTTP',
        Port=80,
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

def create_launch_configuration():
    asg_client = boto3.client('autoscaling')
    
    # Read user-data script from file
    user_data_script = read_user_data_script()
    asg_client.create_launch_configuration(
        LaunchConfigurationName=LAUNCH_CONFIG_NAME,
        ImageId=AMI_ID,
        InstanceType=INSTANCE_TYPE,
        KeyName=KEY_NAME,
        UserData=user_data_script
    )
    print(f"Created Launch Configuration: {LAUNCH_CONFIG_NAME}")

def delete_launch_configuration():
    asg_client = boto3.client('autoscaling')
    asg_client.delete_launch_configuration(
        LaunchConfigurationName=LAUNCH_CONFIG_NAME
    )
    print(f"Deleted Launch Configuration: {LAUNCH_CONFIG_NAME}")

def create_asg(target_group_arn):
    asg_client = boto3.client('autoscaling')
    # Create the launch configuration first
    create_launch_configuration()
    # Create the Auto Scaling Group and associate it with the target group
    asg_client.create_auto_scaling_group(
        AutoScalingGroupName=ASG_NAME,
        LaunchConfigurationName=LAUNCH_CONFIG_NAME,
        MinSize=1,
        MaxSize=5,
        DesiredCapacity=2,
        VPCZoneIdentifier=",".join(VPC_SUBNETS),
        TargetGroupARNs=[target_group_arn]
    )
    print(f"Created Auto Scaling Group: {ASG_NAME}")

def delete_asg():
    asg_client = boto3.client('autoscaling')
    asg_client.delete_auto_scaling_group(
        AutoScalingGroupName=ASG_NAME,
        ForceDelete=True
    )
    print(f"Deleted Auto Scaling Group: {ASG_NAME}")
    # Wait a bit to ensure ASG deletion before deleting the launch configuration
    time.sleep(10)
    delete_launch_configuration()

def delete_listener(listener_arn):
    elbv2 = boto3.client('elbv2')
    elbv2.delete_listener(ListenerArn=listener_arn)
    print(f"Deleted Listener: {listener_arn}")

def delete_target_group(target_group_arn):
    elbv2 = boto3.client('elbv2')
    elbv2.delete_target_group(TargetGroupArn=target_group_arn)
    print(f"Deleted Target Group: {target_group_arn}")

def delete_alb(alb_arn):
    elbv2 = boto3.client('elbv2')
    elbv2.delete_load_balancer(LoadBalancerArn=alb_arn)
    print("Deleted ALB")

if __name__ == '__main__':
    # Create ALB, Target Group, and Listener
    alb_arn, dns_name = create_alb()
    target_group_arn = create_target_group()
    listener_arn = create_listener(alb_arn, target_group_arn)
    
    # Create Auto Scaling Group with instances that register to the target group
    create_asg(target_group_arn)
    
    # Print the ALB HTTP address
    print("\nALB HTTP Address:", f"http://{dns_name}")
    
    # Wait for testing, then clean up resources.
    input("\nPress Enter to delete all resources...")
    
    # Delete ASG, Listener, Target Group, and ALB
    delete_asg()
    delete_listener(listener_arn)
    delete_target_group(target_group_arn)
    delete_alb(alb_arn)
