package agent.tool_access

import rego.v1

user_id(subject) := split(subject, ":")[1]

tool := data.tools[input.resource]
user := data.users[user_id(input.subject)]

has_required_capabilities if {
  required := object.get(tool, "required_capabilities", [])
  user_caps := object.get(user, "capabilities", [])
  every cap in required { cap in user_caps }
}

allow if {
  input.action == "tool.invoke"
  tool
  user
  has_required_capabilities
}

reason := "tool access denied" if {
  not allow
}

reason := "tool access allowed" if {
  allow
}
