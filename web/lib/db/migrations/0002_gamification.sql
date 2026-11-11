CREATE TABLE "user_card" (
	"user_id" text NOT NULL,
	"card_id" text NOT NULL,
	"kind" text NOT NULL,
	"display" text NOT NULL,
	"position" text NOT NULL,
	"overall" integer NOT NULL,
	"count" integer DEFAULT 1 NOT NULL,
	"first_drafted_at" timestamp DEFAULT now() NOT NULL,
	CONSTRAINT "user_card_user_id_card_id_pk" PRIMARY KEY("user_id","card_id")
);
--> statement-breakpoint
ALTER TABLE "achievement" ADD COLUMN "category" text NOT NULL;--> statement-breakpoint
ALTER TABLE "achievement" ADD COLUMN "icon" text NOT NULL;--> statement-breakpoint
ALTER TABLE "achievement" ADD COLUMN "hidden" integer DEFAULT 0 NOT NULL;--> statement-breakpoint
ALTER TABLE "user_card" ADD CONSTRAINT "user_card_user_id_user_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."user"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
CREATE INDEX "user_card_user_idx" ON "user_card" USING btree ("user_id");