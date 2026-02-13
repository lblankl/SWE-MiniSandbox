from swesmith.bug_gen.adapters.golang import get_entities_from_file_go
from swesmith.bug_gen.procedural.golang.remove import (
    RemoveLoopModifier,
    RemoveConditionalModifier,
    RemoveAssignModifier,
)
import random


def test_remove_loop(test_file_go_caddy_listeners):
    entities = []
    get_entities_from_file_go(entities, test_file_go_caddy_listeners)
    pm = RemoveLoopModifier(likelihood=1.0)

    # Set a fixed random seed for reproducible test results
    pm.rand = random.Random(42)

    entities = [x for x in entities if pm.can_change(x)]
    assert len(entities) == 4

    test_entity = entities[0]
    modified = pm.modify(test_entity)
    expected = """func (na NetworkAddress) ListenAll(ctx context.Context, config net.ListenConfig) ([]any, error) {\n\tvar listeners []any\n\tvar err error\n\n\t// if one of the addresses has a failure, we need to close\n\t// any that did open a socket to avoid leaking resources\n\tdefer func() {\n\t\tif err == nil {\n\t\t\treturn\n\t\t}\n\t\t\n\t}()\n\n\t// an address can contain a port range, which represents multiple addresses;\n\t// some addresses don't use ports at all and have a port range size of 1;\n\t// whatever the case, iterate each address represented and bind a socket\n\t\n\n\treturn listeners, nil\n}"""
    assert modified.rewrite == expected


def test_remove_conditional(test_file_go_caddy_listeners):
    entities = []
    get_entities_from_file_go(entities, test_file_go_caddy_listeners)
    pm = RemoveConditionalModifier(likelihood=1.0)

    # Set a fixed random seed for reproducible test results
    pm.rand = random.Random(42)

    entities = [x for x in entities if pm.can_change(x)]
    assert len(entities) == 16

    test_entity = entities[0]
    modified = pm.modify(test_entity)
    expected = """func (na NetworkAddress) ListenAll(ctx context.Context, config net.ListenConfig) ([]any, error) {\n\tvar listeners []any\n\tvar err error\n\n\t// if one of the addresses has a failure, we need to close\n\t// any that did open a socket to avoid leaking resources\n\tdefer func() {\n\t\t\n\t\tfor _, ln := range listeners {\n\t\t\t\n\t\t}\n\t}()\n\n\t// an address can contain a port range, which represents multiple addresses;\n\t// some addresses don't use ports at all and have a port range size of 1;\n\t// whatever the case, iterate each address represented and bind a socket\n\tfor portOffset := uint(0); portOffset < na.PortRangeSize(); portOffset++ {\n\t\tselect {\n\t\tcase <-ctx.Done():\n\t\t\treturn nil, ctx.Err()\n\t\tdefault:\n\t\t}\n\n\t\t// create (or reuse) the listener ourselves\n\t\tvar ln any\n\t\tln, err = na.Listen(ctx, portOffset, config)\n\t\t\n\t\tlisteners = append(listeners, ln)\n\t}\n\n\treturn listeners, nil\n}"""
    assert modified.rewrite == expected


def test_remove_assign(test_file_go_caddy_listeners):
    entities = []
    get_entities_from_file_go(entities, test_file_go_caddy_listeners)
    pm = RemoveAssignModifier(likelihood=1.0)

    # Set a fixed random seed for reproducible test results
    pm.rand = random.Random(42)

    entities = [x for x in entities if pm.can_change(x)]
    assert len(entities) == 14

    test_entity = entities[0]
    modified = pm.modify(test_entity)
    expected = """func (na NetworkAddress) ListenAll(ctx context.Context, config net.ListenConfig) ([]any, error) {\n\tvar listeners []any\n\tvar err error\n\n\t// if one of the addresses has a failure, we need to close\n\t// any that did open a socket to avoid leaking resources\n\tdefer func() {\n\t\tif err == nil {\n\t\t\treturn\n\t\t}\n\t\tfor _, ln := range listeners {\n\t\t\tif ; ok {\n\t\t\t\tcl.Close()\n\t\t\t}\n\t\t}\n\t}()\n\n\t// an address can contain a port range, which represents multiple addresses;\n\t// some addresses don't use ports at all and have a port range size of 1;\n\t// whatever the case, iterate each address represented and bind a socket\n\tfor ; portOffset < na.PortRangeSize();  {\n\t\tselect {\n\t\tcase <-ctx.Done():\n\t\t\treturn nil, ctx.Err()\n\t\tdefault:\n\t\t}\n\n\t\t// create (or reuse) the listener ourselves\n\t\tvar ln any\n\t\t\n\t\tif err != nil {\n\t\t\treturn nil, err\n\t\t}\n\t\t\n\t}\n\n\treturn listeners, nil\n}"""
    assert modified.rewrite == expected
