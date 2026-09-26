# ADR-0007: Collections as Postgres schemas

**Status:** accepted · **Decided by:** project owner
**Context.** Multiple tenants or datasets must not see each other's data; columns must be identical.

**Decision.** One schema per collection (`col_<name>`) created from the same `schema.sql`; a
`public.collections` registry; the collection is selected per connection via `search_path`,
reset on release. Names are strictly validated (they become identifiers).

**Consequences.** Hard isolation (tested), and existing data migrated with `ALTER TABLE … SET SCHEMA`
without re-embedding. No cross-collection search; no delete yet.
