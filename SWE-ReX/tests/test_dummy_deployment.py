from swerex.deployment.dummy import DummyDeployment
from swerex.runtime.abstract import BashAction, CloseBashSessionRequest, CreateBashSessionRequest


async def test_dummy_deployment():
    deployment = DummyDeployment()
    await deployment.start()
    assert deployment.runtime is not None
    await deployment.runtime.create_session(CreateBashSessionRequest())
    await deployment.runtime.run_in_session(BashAction(command="echo hello"))
    await deployment.runtime.close_session(CloseBashSessionRequest())
    assert await deployment.is_alive()
    await deployment.stop()
