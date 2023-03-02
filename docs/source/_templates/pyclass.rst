{{ fullname }}
{{ underline }}

.. currentmodule:: {{ module }}

{%+ if objtype == 'class' -%}
.. autoclass:: {{ objname }}
   :members:
   :special-members: __init__
{% endif %}

{%- if objtype == 'function' -%}
.. autofunction:: {{ objname }}
{% endif %}
