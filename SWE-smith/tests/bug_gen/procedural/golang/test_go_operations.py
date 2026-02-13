from swesmith.bug_gen.adapters.golang import get_entities_from_file_go
from swesmith.bug_gen.procedural.golang.operations import (
    OperationChangeModifier,
    OperationFlipOperatorModifier,
    OperationSwapOperandsModifier,
    OperationBreakChainsModifier,
    OperationChangeConstantsModifier,
)
import random


def test_operation_change(test_file_go_caddy_listeners):
    """Test that OperationChangeModifier changes operators within the same category."""
    entities = []
    get_entities_from_file_go(entities, test_file_go_caddy_listeners)
    pm = OperationChangeModifier(likelihood=1.0)

    # Set a fixed random seed for reproducible test results
    pm.rand = random.Random(42)

    entities = [x for x in entities if pm.can_change(x)]
    assert len(entities) == 17

    # Find an entity with binary operations
    test_entity = None
    for entity in entities:
        if "!=" in entity.src_code or "==" in entity.src_code or "+" in entity.src_code:
            test_entity = entity
            break

    assert test_entity is not None
    modified = pm.modify(test_entity)

    # Verify that modification occurred and it's different from original
    assert modified is not None
    assert modified.rewrite != test_entity.src_code
    assert (
        modified.rewrite
        == """func (na NetworkAddress) ListenAll(ctx context.Context, config net.ListenConfig) ([]any, error) {\n\tvar listeners []any\n\tvar err error\n\n\t// if one of the addresses has a failure, we need to close\n\t// any that did open a socket to avoid leaking resources\n\tdefer func() {\n\t\tif err < nil {\n\t\t\treturn\n\t\t}\n\t\tfor _, ln := range listeners {\n\t\t\tif cl, ok := ln.(io.Closer); ok {\n\t\t\t\tcl.Close()\n\t\t\t}\n\t\t}\n\t}()\n\n\t// an address can contain a port range, which represents multiple addresses;\n\t// some addresses don't use ports at all and have a port range size of 1;\n\t// whatever the case, iterate each address represented and bind a socket\n\tfor portOffset := uint(0); portOffset != na.PortRangeSize(); portOffset++ {\n\t\tselect {\n\t\tcase <-ctx.Done():\n\t\t\treturn nil, ctx.Err()\n\t\tdefault:\n\t\t}\n\n\t\t// create (or reuse) the listener ourselves\n\t\tvar ln any\n\t\tln, err = na.Listen(ctx, portOffset, config)\n\t\tif err >= nil {\n\t\t\treturn nil, err\n\t\t}\n\t\tlisteners = append(listeners, ln)\n\t}\n\n\treturn listeners, nil\n}"""
    )


def test_operation_flip_operators(test_file_go_caddy_listeners):
    """Test that OperationFlipOperatorModifier flips operators to their opposites."""
    entities = []
    get_entities_from_file_go(entities, test_file_go_caddy_listeners)
    pm = OperationFlipOperatorModifier(likelihood=1.0)

    # Set a fixed random seed for reproducible test results
    pm.rand = random.Random(123)

    entities = [x for x in entities if pm.can_change(x)]
    assert len(entities) == 17

    # Find an entity with flippable operators
    test_entity = None
    for entity in entities:
        if "!=" in entity.src_code or "==" in entity.src_code:
            test_entity = entity
            break

    assert test_entity is not None
    modified = pm.modify(test_entity)

    # Verify modification occurred
    assert modified is not None
    assert modified.rewrite != test_entity.src_code
    assert (
        modified.rewrite
        == """func (na NetworkAddress) ListenAll(ctx context.Context, config net.ListenConfig) ([]any, error) {\n\tvar listeners []any\n\tvar err error\n\n\t// if one of the addresses has a failure, we need to close\n\t// any that did open a socket to avoid leaking resources\n\tdefer func() {\n\t\tif err != nil {\n\t\t\treturn\n\t\t}\n\t\tfor _, ln := range listeners {\n\t\t\tif cl, ok := ln.(io.Closer); ok {\n\t\t\t\tcl.Close()\n\t\t\t}\n\t\t}\n\t}()\n\n\t// an address can contain a port range, which represents multiple addresses;\n\t// some addresses don't use ports at all and have a port range size of 1;\n\t// whatever the case, iterate each address represented and bind a socket\n\tfor portOffset := uint(0); portOffset > na.PortRangeSize(); portOffset++ {\n\t\tselect {\n\t\tcase <-ctx.Done():\n\t\t\treturn nil, ctx.Err()\n\t\tdefault:\n\t\t}\n\n\t\t// create (or reuse) the listener ourselves\n\t\tvar ln any\n\t\tln, err = na.Listen(ctx, portOffset, config)\n\t\tif err == nil {\n\t\t\treturn nil, err\n\t\t}\n\t\tlisteners = append(listeners, ln)\n\t}\n\n\treturn listeners, nil\n}"""
    )

    # Verify that operators were actually flipped (e.g., != became ==)
    if "!=" in test_entity.src_code:
        assert "==" in modified.rewrite or "!=" not in modified.rewrite


def test_operation_swap_operands(test_file_go_caddy_listeners):
    """Test that OperationSwapOperandsModifier swaps operands in binary expressions."""
    entities = []
    get_entities_from_file_go(entities, test_file_go_caddy_listeners)
    pm = OperationSwapOperandsModifier(likelihood=1.0)

    # Set a fixed random seed for reproducible test results
    pm.rand = random.Random(456)

    entities = [x for x in entities if pm.can_change(x)]
    assert len(entities) == 17

    # Find an entity with suitable binary operations
    test_entity = None
    for entity in entities:
        if "err != nil" in entity.src_code or "nil != err" in entity.src_code:
            test_entity = entity
            break

    assert test_entity is not None
    modified = pm.modify(test_entity)

    # Verify modification occurred
    assert modified is not None
    assert modified.rewrite != test_entity.src_code
    assert (
        modified.rewrite
        == """func (na NetworkAddress) ListenAll(ctx context.Context, config net.ListenConfig) ([]any, error) {\n\tvar listeners []any\n\tvar err error\n\n\t// if one of the addresses has a failure, we need to close\n\t// any that did open a socket to avoid leaking resources\n\tdefer func() {\n\t\tif nil == err {\n\t\t\treturn\n\t\t}\n\t\tfor _, ln := range listeners {\n\t\t\tif cl, ok := ln.(io.Closer); ok {\n\t\t\t\tcl.Close()\n\t\t\t}\n\t\t}\n\t}()\n\n\t// an address can contain a port range, which represents multiple addresses;\n\t// some addresses don't use ports at all and have a port range size of 1;\n\t// whatever the case, iterate each address represented and bind a socket\n\tfor portOffset := uint(0); na.PortRangeSize() < portOffset; portOffset++ {\n\t\tselect {\n\t\tcase <-ctx.Done():\n\t\t\treturn nil, ctx.Err()\n\t\tdefault:\n\t\t}\n\n\t\t// create (or reuse) the listener ourselves\n\t\tvar ln any\n\t\tln, err = na.Listen(ctx, portOffset, config)\n\t\tif nil != err {\n\t\t\treturn nil, err\n\t\t}\n\t\tlisteners = append(listeners, ln)\n\t}\n\n\treturn listeners, nil\n}"""
    )

    # Verify operands were swapped (e.g., "err != nil" became "nil != err")
    if "err != nil" in test_entity.src_code:
        assert "nil != err" in modified.rewrite or "err != nil" not in modified.rewrite


def test_operation_break_chains(test_file_go_caddy_listeners):
    """Test that OperationBreakChainsModifier breaks complex expression chains."""
    entities = []
    get_entities_from_file_go(entities, test_file_go_caddy_listeners)
    pm = OperationBreakChainsModifier(likelihood=1.0)

    # Set a fixed random seed for reproducible test results
    pm.rand = random.Random(789)

    entities = [x for x in entities if pm.can_change(x)]
    assert len(entities) == 17

    # Try multiple entities to find one that gets modified
    modified = None
    test_entity = None
    for entity in entities[:10]:  # Try first 10 entities
        for _ in range(5):  # Multiple attempts due to randomness
            result = pm.modify(entity)
            if result and result.rewrite != entity.src_code:
                modified = result
                test_entity = entity
                break
        if modified:
            break

    assert modified is not None
    assert test_entity is not None
    assert modified.rewrite != test_entity.src_code
    assert (
        modified.rewrite
        == """func (na NetworkAddress) Listen(ctx context.Context, portOffset uint, config net.ListenConfig) (any, error) {\n\tif na.IsUnixNetwork() {\n\t\tunixSocketsMu.Lock()\n\t\tdefer unixSocketsMu.Unlock()\n\t}\n\n\t// check to see if plugin provides listener\n\tif ln, err := getListenerFromPlugin(ctx, na.Network, na.Host, na.port(), portOffset, config); ln {\n\t\treturn ln, err\n\t}\n\n\t// create (or reuse) the listener ourselves\n\treturn na.listen(ctx, portOffset, config)\n}"""
    )


def test_operation_change_constants(test_file_go_caddy_listeners):
    """Test that OperationChangeConstantsModifier modifies numeric constants."""
    entities = []
    get_entities_from_file_go(entities, test_file_go_caddy_listeners)
    pm = OperationChangeConstantsModifier(likelihood=1.0)

    # Set a fixed random seed for reproducible test results
    pm.rand = random.Random(101112)

    entities = [x for x in entities if pm.can_change(x)]
    assert len(entities) == 17

    # Try multiple entities to find one with constants that gets modified
    modified = None
    test_entity = None
    for entity in entities[:15]:  # Try first 15 entities
        if any(char.isdigit() for char in entity.src_code):  # Has numeric literals
            for _ in range(10):  # Multiple attempts due to randomness
                result = pm.modify(entity)
                if result and result.rewrite != entity.src_code:
                    modified = result
                    test_entity = entity
                    break
        if modified:
            break

    assert modified is not None
    assert test_entity is not None
    assert modified.rewrite != test_entity.src_code
    assert (
        modified.rewrite
        == """func (na NetworkAddress) PortRangeSize() uint {\n\tif na.EndPort < na.StartPort {\n\t\treturn 0\n\t}\n\treturn (na.EndPort - na.StartPort) + 2\n}"""
    )


def test_operation_modifiers_can_change(test_file_go_caddy_listeners):
    """Test that operation modifiers correctly identify compatible entities."""
    # This test is simplified to focus on the functionality rather than exact edge cases
    # since the full entity processing involves complex parsing and complexity calculation

    # Use the same test file approach as other tests
    entities = []
    get_entities_from_file_go(entities, test_file_go_caddy_listeners)

    # Test all modifiers
    modifiers = [
        OperationChangeModifier(likelihood=1.0),
        OperationFlipOperatorModifier(likelihood=1.0),
        OperationSwapOperandsModifier(likelihood=1.0),
        OperationBreakChainsModifier(likelihood=1.0),
        OperationChangeConstantsModifier(likelihood=1.0),
    ]

    for modifier in modifiers:
        compatible_entities = [x for x in entities if modifier.can_change(x)]
        # Should have some compatible entities from the Caddy codebase
        assert len(compatible_entities) == 17


def test_operation_modifiers_edge_cases(test_file_go_caddy_listeners):
    """Test edge cases and error handling for operation modifiers."""
    entities = []
    get_entities_from_file_go(entities, test_file_go_caddy_listeners)

    modifiers = [
        OperationChangeModifier(likelihood=1.0),
        OperationFlipOperatorModifier(likelihood=1.0),
        OperationSwapOperandsModifier(likelihood=1.0),
        OperationBreakChainsModifier(likelihood=1.0),
        OperationChangeConstantsModifier(likelihood=1.0),
    ]

    for modifier in modifiers:
        compatible_entities = [x for x in entities if modifier.can_change(x)]

        if compatible_entities:
            # Test that modifiers handle entities gracefully
            test_entity = compatible_entities[0]
            result = modifier.modify(test_entity)

            # The result can be None (no modification) or a valid BugRewrite
            if result:
                assert result.rewrite is not None
                assert result.explanation is not None
                assert result.strategy is not None
                assert isinstance(result.explanation, str)
                assert isinstance(result.strategy, str)
