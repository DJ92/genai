package agent.data_scopes

import rego.v1

user_id(subject) := split(subject, ":")[1]
user := data.users.users[user_id(input.subject)]

requested_scopes := object.get(input.context, "request_scopes", ["personal"])
all_scopes := [scope | data.scopes.scopes[scope]]

user_scopes := all_scopes if {
  "admin" in object.get(user, "roles", [])
}

user_scopes := ["personal", "work"] if {
  not "admin" in object.get(user, "roles", [])
}

allowed_scopes := [s | s := requested_scopes[_]; s in user_scopes]

allow if {
  input.action == "data.read"
  user
  count(allowed_scopes) > 0
}

reason := "scope access denied" if {
  not allow
}

reason := "scope access allowed" if {
  allow
}
