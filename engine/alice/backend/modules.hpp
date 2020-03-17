/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "engine/alice/component.hpp"
#include "engine/core/optional.hpp"
#include "third_party/nlohmann/json.hpp"

namespace isaac {
namespace alice {

// Manges component modules
class ModuleManager {
 public:
  // Loads all comonents which are linked into the application itself
  void loadStaticallyLinked();

  // Tries to loads a module at the given location
  std::optional<std::string> loadModule(const std::string& filename);

  // Loads all given modules at the given locations
  void loadModules(const std::vector<std::string>& filenames);

  // Unloads the module at the given location
  void unloadModule(const std::string& name);

  // Create a component of the given type. The component may not be present in multiple modules.
  Component* createComponent(const std::string& type_name) const;

  // Gets a list with the (unique) names of all modules
  std::set<std::string> getModuleNames() const;
  // Gets a list with the (unique) names of all components which where registered
  std::set<std::string> getComponentNames() const;

  // Set base path for module paths
  void setModuleWorkingPath(const std::string& working_path);

  // Adds more paths to search for module shared libraries
  void appendModulePaths(const std::vector<std::string>& paths);

 private:
  using GatherComponentInfoFunction = int(char*, int);
  using CreateComponentFunction = void*(const char*);

  struct Module {
   public:
    ~Module();

    // Opens the module at the given location. As a result all components provided by this module
    // become available for usage.
    void open(const std::vector<std::string>& paths, const std::string& filename,
              const std::string& working_path);
    // Returns true if the module was loaded successfully
    bool isOpen() const { return handle_; }
    // Gets an error message
    const std::string& getErrorMessage() const { return error_message_; }
    // Closes this module. As a result all components provided by this module will become
    // unavailable.
    void close();

    // The name of this module
    const std::string& name() const { return name_; }
    // A list of all components loaded by this module
    const std::set<std::string>& components() const { return components_; }
    // Creates a new component of the given type name (only calls constructor)
    Component* createComponent(const std::string& type_name) const;

   private:
    // Loads the symbol with the given name from the module
    void* loadSymbol(const std::string& name);

    std::string filename_;
    void* handle_ = nullptr;
    std::string name_;
    nlohmann::json component_info_;
    std::set<std::string> components_;
    std::string error_message_;

    GatherComponentInfoFunction* gather_component_info_function_;
    CreateComponentFunction* create_component_function_;
  };

  // Names of components which are statically linked
  std::set<std::string> self_components_;

  // Base path to attempt to load modules with relative paths
  std::string working_path_;
  // Paths to search module shared libraries
  std::vector<std::string> module_paths_{"packages_x86_64", "packages_jetpack43"};

  std::map<std::string, std::unique_ptr<Module>> modules_;
  mutable std::mutex mutex_;
};

}  // namespace alice
}  // namespace isaac
