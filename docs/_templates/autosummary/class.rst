..
  class.rst

{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
    :members:
    :show-inheritance:
    :inherited-members:
    :undoc-members:

    {% block methods %}
        {% if methods %}
            .. rubric:: {{ _('New methods & Specialized methods') }}

            .. autosummary::
                :nosignatures:
                {% for item in all_methods %}
                    {% if item not in inherited_members %}
                    ~{{ name }}.{{ item }}
                    {% endif %}
                {%- endfor %}
        {% endif %}
    {% endblock %}

    {% block inherited %}
        {% if inherited_members.intersection(methods) %}
            .. rubric:: {{ _('Methods inherited from parent') }}

            .. autosummary::
                :nosignatures:
                {% for item in inherited_members.intersection(methods) %}
                    {% if item in inherited_members %}
                    ~{{ name }}.{{ item }}
                    {% endif %}
                {%- endfor %}
        {% endif %}
    {% endblock %}

    {% block attributes %}
        {% if attributes %}
            .. rubric:: {{ _('Attributes and properties') }}

            .. autosummary::
                :nosignatures:
                {% for item in attributes %}
                    ~{{ name }}.{{ item }}
                {%- endfor %}
        {% endif %}
    {% endblock %}

    {% block details %}
            .. rubric {{ _('Details') }}
    {% endblock %}
