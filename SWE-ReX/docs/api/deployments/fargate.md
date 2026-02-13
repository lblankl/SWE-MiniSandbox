!!! warning
    This deployment is currently in alpha stage. Expect breaking changes.

## AWS Resource Cleanup

The Fargate deployment creates several AWS resources that persist even after your deployment stops. These resources include:

- ECS Clusters
- Task Definitions 
- Security Groups
- IAM Roles

All resources created by the Fargate deployment are tagged with `origin=swe-rex-deployment-auto` for tracking purposes.

### Cleaning Up Resources

To clean up all AWS resources created by the Fargate deployment, you can use the built-in teardown utility:

```bash
python -m swerex.utils.aws_teardown
```

This utility will:

1. Preview all resources tagged with `origin=swe-rex-deployment-auto`
2. Request confirmation before deletion
3. Delete resources in the correct order to handle dependencies
4. Provide status updates during the cleanup process

!!! tip
    It's recommended to run the teardown utility periodically to avoid accumulating unused AWS resources, which may incur costs.
    Running the fargate deployment again will recreate the necessary resources on the fly.

::: swerex.deployment.fargate.FargateDeployment
