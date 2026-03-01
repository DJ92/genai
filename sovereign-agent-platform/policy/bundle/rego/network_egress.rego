package agent.network_egress

import rego.v1

tool_name := object.get(input.context, "tool_name", "")
tool := data.tools[tool_name]
host := input.resource

allow if {
  input.action == "network.egress"
  tool
  object.get(tool, "requires_network_egress", false)
  host in object.get(tool, "allowed_hosts", [])
}

reason := "network egress denied" if {
  not allow
}

reason := "network egress allowed" if {
  allow
}
