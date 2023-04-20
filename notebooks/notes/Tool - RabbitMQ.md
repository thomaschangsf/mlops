<-- Theory/Concept

	* Pattern 1: Worker Queue
		- all consumers shares a queue

		- Pub --> Q --> {consumer|worker}_n


	* Pattern 2a: Publisher Subscriber
		- each consumer has a queue

		- Flow: Publisher --> Exchange --> { (Q -> consumer) }_n

		- consumers all receive the same msg!  
			good for chat system, or friends following each other

	* Pattern 2b: Routing
		- Flow: Publisher --> Exchange(type=direct) --> { filter_route( (Q -> consumer) ) }_n

		- consumer/receiver can receive msg selectively


	* Pattern 2c: Topic
		- Flow: Publisher --> Exchange(type=topic) --> { filter_route( (Q -> consumer) ) }_n

		- consumer can receive msg selectivley based on topics, based on pattern matching of topic 



<-- Experiences
	* Duplicate records:
		what are the constraints of the schema

			primary (composite) keys
		  	constraints
		  		types: https://www.ibm.com/docs/en/ias?topic=constraints-types

		  		- NOT NULL constraints prevent null values from being entered into a column.
				
				- Unique constraints ensure that the values in a set of columns are unique and not null for all rows in the table. The columns specified in a unique constraint must be defined as NOT NULL. The database manager uses a unique index to enforce the uniqueness of the key during changes to the columns of the unique constraint.

				- Primary key constraints
				You can use primary key and foreign key constraints to define relationships between tables.
				
				- (Table) Check constraints
				A check constraint (also referred to as a table check constraint) is a database rule that specifies the values allowed in one or more columns of every row of a table. Specifying check constraints is done through a restricted form of a search condition.
				Foreign key (referential) constraints
				Foreign key constraints (also known as referential constraints or referential integrity constraints) enable definition of required relationships between and within tables.
				
				- Informational constraints
				An informational constraint is a constraint attribute that can be used by the SQL compiler to improve the access to data. Informational constraints are not enforced by the database manager, and are not used for additional verification of data; rather, they are used to improve query performance.

		NOSQL has limited ability of constraints, such as check constraints.  The burden shifts to the application developer.


		
