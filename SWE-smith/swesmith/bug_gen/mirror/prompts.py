RECOVERY_PROMPT = """You are given the source code of a file and a corresponding diff patch that reflects changes made to this file.
Your task is to rewrite the entire source code while reversing the changes indicated by the diff patch.
That is, if a line was added in the diff, remove it; if a line was removed, add it back; and if a line was modified, restore it to its previous state.

DO NOT MAKE ANY OTHER CHANGES TO THE SOURCE CODE. If a line was not explicitly added or removed in the diff, it should remain unchanged in the output.

INPUT:
<source_code>
Source code will be provided here.
</source_code>

<diff_patch>
Diff patch will be provided here.
</diff_patch>

OUTPUT:
The fully rewritten source code, after undoing all changes specified in the diff.
The output should be valid Python code.
"""

DEMO_PROMPT = """Demonstration:

INPUT:
<source_code>
def greet(name):
    print(f"Hi, {name}! How's it going?")
    print("Even though this line is not in the diff, it should remain unchanged.")

def farewell(name):
    print(f"Goodbye, {name}!")
</source_code>

<diff_patch>
diff --git a/greet.py b/greet.py
index 1234567..7654321 100644
--- a/greet.py
+++ b/greet.py
@@ -1,4 +1,4 @@
 def greet(name):
-    print(f"Hello, {name}! How are you?")
+    print(f"Hi, {name}! How's it going?")

 def farewell(name):
     print(f"Goodbye, {name}!")
</diff_patch>
</input>

OUTPUT:
def greet(name):
    print(f"Hello, {name}! How are you?")
    print("Even though this line is not in the diff, it should remain unchanged.")

def farewell(name):
    print(f"Goodbye, {name}!")
"""

TASK_PROMPT = """Task:

INPUT:
<source_code>
{}
</source_code>

<diff_patch>
{}
</diff_patch>
</input>

NOTES:
- As a reminder, DO NOT MAKE ANY OTHER CHANGES TO THE SOURCE CODE. If a line was not explicitly added or removed in the diff, it should remain unchanged in the output.
- Only make changes based on lines that were:
    * Added (have a + in front of them)
    * Removed (have a - in front of them)
- DO NOT PROVIDE ANY TEXT ASIDE FROM THE REWRITTEN FILE. ANSWER WITH ONLY THE REWRITTEN CODE.

OUTPUT:"""
