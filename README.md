<p align="center">
  <img src="https://github.com/ziyadedher/decuen/blob/develop/.github/images/logo.png?raw=true">
</p>

# Decuen
![GitHub last commit](https://img.shields.io/github/last-commit/ziyadedher/decuen)
![Travis (.com)](https://img.shields.io/travis/com/ziyadedher/decuen)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/ziyadedher/decuen)
![PyPI](https://img.shields.io/pypi/v/decuen)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/decuen)

[Decuen](https://github.com/ziyadedher/decuen) (dee-kew-en) is a reinforcement learning framework built in modern Python for experimentation, ease-of-use, modularity, and extensibility. Containing building blocks for modern reinforcement learning systems and modular implementations of common reinforcement learning algorithms, Decuen provides a platform for rapid iterative experimentation of new reinforcement learning systems.

Decuen has a robust Python API with type annotations (see the Python [typing](https://docs.python.org/3/library/typing.html) module) which provides developers and users with a clear view of API contracts that can accelerate debugging on both the developer and user side.

Decuen development and design is lead by [Ziyad Edher](https://github.com/ziyadedher) ![GitHub followers](https://img.shields.io/github/followers/ziyadedher?style=social) ![Twitter Follow](https://img.shields.io/twitter/follow/ziyadedher?style=social).

![GitHub forks](https://img.shields.io/github/forks/ziyadedher/decuen?style=social)
![GitHub stars](https://img.shields.io/github/stars/ziyadedher/decuen?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/ziyadedher/decuen?style=social)


## Principles
The development of Decuen follows a small but effective set of guiding principles that attempts to span and encapsulate our vision of what Decuen is to be:
* **Ease of use.** Decuen provides an API that follows existing standards, terminology, and intuitions popularized in modern reinforcement learning literature. Following these standards allows the usage of Decuen to come as second-nature to developers or researchers already familiar with reinforcement learning. Development of Decuen follows this standard to reduce the barrier-of-entry to using the framework.
* **Extensibility.** Decuen is intended for quick, iterative development and experimentation of reinforcement learning systems, and as such the ability to easily extend Decuen source code is of utmost importance. Decuen achieves this by providing well-documented and typed source code with thoughtfully designed interfaces and abstractions.
* **Modularity.** Decuen strives to be a tool for developers and researchers alike to be able to create reinforcement learning mechanisms without reinventing the wheel. Decuen implements all algorithms in a modular "building blocks" fashion to allow the plug-and-play of different modules in the larger system. This allows the independent design and development of new components to be tested as replacements for modules in current reinforcement learning systems.
* **Transparency.** Decuen strives to be as transparent as possible in its interfaces and abstractions to facilitate as many dimensions of change as possible. Abstractions, interfaces, and implementations are not closed-off and tend to provide as much information about internal processes as possible while abiding by our standard of modularity and extensibility.


## Installation
#### Basic Installation
Package is not yet released on PyPI for basic release installation.


#### Source Installation
Installing `decuen` from source is very simple given the framework is entirely written in modern Python:
0. Be in the environment you would like to install `decuen` in.
1. Clone the repository: `git clone https://github.com/ziyadedher/decuen`.
2. Install `decuen` into your environment: `cd decuen && pip install .`.


## Usage
After installing `decuen`, the framework is completely ready to be used. Detailed usage instructions, tutorials, examples, and more can be found in the [Decuen Wiki](https://github.com/ziyadedher/decuen/wiki).


## Development
After installing `decuen` from source in your environment, there is a step to perform before the project is ready to be developed on:
1. Reinstall `decuen` in editable mode with linting and testing libraries: `pip install -e '.[lint, test]'`.

Now the project is ready to be developed. Note that in order to make sure that the CI/CD pipeline passes on your merge request, verify that our linter `prospector` does not produce any errors or warnings. You can also use `tox` to run the linter.
