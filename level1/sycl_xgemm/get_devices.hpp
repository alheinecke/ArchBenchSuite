#pragma once

#include <sycl/sycl.hpp>
#include <vector>

namespace helper {

template <typename Selector = decltype(sycl::default_selector_v)>
auto get_devices(Selector&& selector = sycl::default_selector_v, bool split_numa = true) {
  sycl::platform p(std::forward<Selector>(selector));
  auto root_devices = p.get_devices();

  std::vector<sycl::device> devices;

  if (!split_numa) {
    return root_devices;
  } else {
    try {
      for (auto &&root_device : root_devices) {
        auto subdevices = root_device.create_sub_devices<
            sycl::info::partition_property::partition_by_affinity_domain>(
            sycl::info::partition_affinity_domain::numa);

        for (auto &&subdevice : subdevices) {
          devices.push_back(subdevice);
        }
      }
    } catch(...) {
      return root_devices;
    }
    return devices;
  }
}

} // end helper
