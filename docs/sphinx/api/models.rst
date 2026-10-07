Models
======

Public helpers use the default registry root. Create ``Registry(root=...)``
when you need a separate cache.

.. autofunction:: manuscript.models.info

Returns metadata for a key; ``model_class`` optionally validates the consuming
class. An unknown key triggers a catalog refresh before raising ``ValueError``.

.. autofunction:: manuscript.models.refresh

Refreshes configured sources and returns the effective catalog.

.. autofunction:: manuscript.models.add_registry

Adds a source for this process and optionally persists it. Call ``refresh()``
after changing sources.

.. autofunction:: manuscript.models.resolve

Returns a mapping of artifact roles to local paths. With ``artifact=None``, fetches
required artifacts; otherwise fetches the requested role. A declared license is
included. ``force_download=True`` bypasses reuse of existing artifact files.

.. autofunction:: manuscript.models.download

Alias of ``resolve()`` with the same arguments and return value.

.. autofunction:: manuscript.models.path

Returns the bundle directory without downloading artifacts. Looking up its
metadata may refresh the catalog; the directory and manifest may be created.

.. autofunction:: manuscript.models.list_installed

.. autofunction:: manuscript.models.verify

Returns a mapping from artifact role to boolean verification result.

.. autofunction:: manuscript.models.is_installed

Returns whether all required verification checks pass.

.. autofunction:: manuscript.models.remove

.. autoclass:: manuscript.models.Registry
   :no-members:

For source priority, compatibility warnings and cache layout, see
:doc:`../model_registry`.
