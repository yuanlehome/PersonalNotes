#pragma once

#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "variable.h"

/**
 * @brief Scope that manage all variables.
 *
 * Scope is an association of a name to Variable. All variables belong to
 * Scope. You need to specify a scope to run a Net, i.e., `net.Run(&scope)`.
 * One net can run in different scopes and update different variable in the
 * scope.
 */
class Scope {
 public:
  Scope() {}
  ~Scope() { };

  /// Create a variable with given name if it doesn't exist.
  /// Caller doesn't own the returned Variable.
  Variable* Var(const std::string& name) {
    Variable* ret = nullptr;
    ret = VarInternal(name);
    return ret;
  }

  /// Find a variable in the scope or any of its ancestors.  Returns
  /// nullptr if cannot find.
  /// Caller doesn't own the returned Variable.
  Variable* FindVar(const std::string& name) const {
    auto it = vars_.find(name);
    if (it != vars_.end()) {
      return it->second.get();
    }
    return nullptr;
  }

  // Get a variable in the scope or any of its ancestors. Enforce
  /// the returned Variable is not nullptr
  Variable* GetVar(const std::string& name) const {
    auto* var = FindVar(name);
    ENFORCE_NOT_NULL(
        var, "Cannot find %s in scope.", name);
    return var;
  }

  // Return the number of variables in scope
  size_t Size() { return vars_.size(); }

 protected:
  mutable std::unordered_map<std::string, std::unique_ptr<Variable>>vars_;

 private:
  Variable* FindVarLocally(const std::string& name) const {
    auto it = vars_.find(name);
    if (it != vars_.end()) {
      return it->second.get();
    }
    return nullptr;
  }

  Variable* VarInternal(const std::string& name) {
    auto* v = FindVarLocally(name);
    if (v != nullptr) return v;
    v = new Variable();
    vars_.emplace(name, std::unique_ptr<Variable>(v));
    return v;
  }

 private:     
  // Disable the copy and assignment                               
  Scope(const Scope&) = delete;           
  Scope(Scope&&) = delete;                
  Scope& operator=(const Scope&) = delete;
  Scope& operator=(Scope&&) = delete;
};