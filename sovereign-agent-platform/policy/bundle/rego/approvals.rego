package agent

import rego.v1
import data.agent.tool_access
import data.agent.data_scopes
import data.agent.network_egress

requires_dangerous_tool_approval if {
  input.action == "tool.invoke"
  tool := data.tools[input.resource]
  object.get(tool, "dangerous", false)
  object.get(input.context, "require_approval_for_dangerous_tools", true)
}

default decision := {
  "decision": "deny",
  "reason": "default deny",
  "constraints": {}
}

decision := {
  "decision": "needs_approval",
  "reason": "dangerous tool requires approval",
  "constraints": {}
} if {
  requires_dangerous_tool_approval
}

decision := {
  "decision": "allow",
  "reason": tool_access.reason,
  "constraints": {
    "allowed_hosts": object.get(data.tools[input.resource], "allowed_hosts", [])
  }
} if {
  input.action == "tool.invoke"
  tool_access.allow
  not requires_dangerous_tool_approval
}

decision := {
  "decision": "deny",
  "reason": tool_access.reason,
  "constraints": {}
} if {
  input.action == "tool.invoke"
  not tool_access.allow
  not requires_dangerous_tool_approval
}

decision := {
  "decision": "allow",
  "reason": data_scopes.reason,
  "constraints": {
    "allowed_scopes": data_scopes.allowed_scopes
  }
} if {
  input.action == "data.read"
  data_scopes.allow
}

decision := {
  "decision": "deny",
  "reason": data_scopes.reason,
  "constraints": {
    "allowed_scopes": data_scopes.allowed_scopes
  }
} if {
  input.action == "data.read"
  not data_scopes.allow
}

decision := {
  "decision": "allow",
  "reason": network_egress.reason,
  "constraints": {
    "allowed_hosts": object.get(data.tools[tool_name], "allowed_hosts", [])
  }
} if {
  input.action == "network.egress"
  tool_name := object.get(input.context, "tool_name", "")
  network_egress.allow
}

decision := {
  "decision": "deny",
  "reason": network_egress.reason,
  "constraints": {}
} if {
  input.action == "network.egress"
  not network_egress.allow
}
